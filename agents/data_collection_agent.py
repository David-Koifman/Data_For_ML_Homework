"""
DataCollectionAgent — Задание 1
Собирает данные из нескольких источников и возвращает унифицированный DataFrame.
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


STAR_MAP = {"One": 0, "Two": 0, "Three": None, "Four": 1, "Five": 1}


class DataCollectionAgent:
    def __init__(self, config: str = "config.yaml"):
        with open(config, "r") as f:
            self.config = yaml.safe_load(f)
        self.task = self.config.get("task", "")

    # ── skill: load_dataset ──────────────────────────────────────────────────
    def load_dataset(self, name: str, source: str = "hf", split: str = "train",
                     sample: int = None, label_map: dict = None, text_col: str = None,
                     config_name: str = None) -> pd.DataFrame:
        if source == "hf":
            from datasets import load_dataset as hf_load
            label = f"{name}/{config_name}" if config_name else name
            print(f"  [load_dataset] Загружаю {label} с HuggingFace...")
            ds = hf_load(name, config_name, split=split) if config_name else hf_load(name, split=split)
            df = ds.to_pandas()
            if sample:
                df = df.sample(n=min(sample, len(df)), random_state=42)
            # Находим колонку с текстом: явно заданная → или автопоиск
            TEXT_CANDIDATES = ["text", "sentence", "review", "content", "comment", "body"]
            if text_col and text_col in df.columns:
                df = df.rename(columns={text_col: "text"})
            elif "text" not in df.columns:
                for candidate in TEXT_CANDIDATES:
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: "text"})
                        break
            # Применяем маппинг меток из config.yaml (если задан)
            if label_map and "label" in df.columns:
                # config.yaml хранит ключи как int или str — нормализуем
                normalized_map = {}
                for k, v in label_map.items():
                    normalized_map[int(k)] = v
                    normalized_map[str(k)] = v
                df["label"] = df["label"].map(normalized_map)
            elif "label" in df.columns and df["label"].dtype in ["int64", "int32"]:
                # fallback: числовые метки → строки как есть
                df["label"] = df["label"].astype(str)
            df["source"] = f"hf:{name}"
            df["collected_at"] = datetime.now().isoformat()
            df = df.dropna(subset=["label"])
            print(f"  [load_dataset] Загружено {len(df)} записей")
            return df[["text", "label", "source", "collected_at"]]
        else:
            raise ValueError(f"Источник '{source}' не поддерживается. Используй 'hf'.")

    # ── skill: scrape ────────────────────────────────────────────────────────
    def scrape(self, url: str, selector: str, pages: int = 1,
               text_selector: str = None, fixed_label: str = None,
               label_from_class: str = None, label_map: dict = None,
               page_param: str = None) -> pd.DataFrame:
        """
        Универсальный скрапер.
        - text_selector: CSS-селектор текста внутри каждого item
        - fixed_label: фиксированная метка для всех записей
        - label_from_class: префикс CSS-класса откуда брать метку (напр. "m-statement--")
        - label_map: маппинг извлечённых меток → positive/negative
        - page_param: параметр пагинации через URL (?page=N), если None — ищем li.next a
        """
        print(f"  [scrape] Скрапинг {url} ({pages} страниц)...")
        records = []
        headers = {"User-Agent": "Mozilla/5.0"}

        for page in range(1, pages + 1):
            # Строим URL страницы
            if page_param:
                current_url = f"{url}?{page_param}={page}" if page > 1 else url
            else:
                current_url = url

            try:
                resp = requests.get(current_url, timeout=10, headers=headers)
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select(selector)

                if not items:
                    break

                for item in items:
                    # Извлекаем текст
                    if text_selector:
                        text_tag = item.select_one(text_selector)
                        if not text_tag:
                            continue
                        text = text_tag.get_text(strip=True)
                    else:
                        # books.toscrape режим
                        title_tag = item.select_one("h3 a")
                        if not title_tag:
                            continue
                        text = title_tag.get("title", title_tag.text.strip())

                    # Извлекаем метку
                    if label_from_class:
                        # Берём из CSS-класса: "m-statement--false" → "false"
                        # Ищем класс чей суффикс есть в label_map
                        classes = item.get("class", [])
                        label = None
                        for c in classes:
                            if c.startswith(label_from_class) and c != label_from_class:
                                raw_label = c.replace(label_from_class, "")
                                mapped = (label_map or {}).get(raw_label)
                                if mapped is not None:
                                    label = mapped
                                    break
                        if label is None:
                            continue  # не в маппинге — пропускаем
                    elif fixed_label:
                        label = fixed_label
                    else:
                        # books.toscrape режим
                        rating_tag = item.select_one("p.star-rating")
                        if not rating_tag:
                            continue
                        rating_word = rating_tag.get("class", ["", ""])[1]
                        lbl = STAR_MAP.get(rating_word)
                        if lbl is None:
                            continue
                        label = "positive" if lbl == 1 else "negative"

                    if not text:
                        continue
                    records.append({
                        "text": text,
                        "label": label,
                        "source": f"scrape:{url}",
                        "collected_at": datetime.now().isoformat()
                    })

                # Пагинация через li.next a (универсальный)
                if not page_param:
                    next_btn = soup.select_one("li.next a")
                    if next_btn and page < pages:
                        next_href = next_btn["href"]
                        current_url = urljoin(current_url, next_href)
                    else:
                        break

            except Exception as e:
                print(f"  [scrape] Ошибка на странице {page}: {e}")
                break

        df = pd.DataFrame(records)
        print(f"  [scrape] Собрано {len(df)} записей")
        return df

    # ── skill: fetch_api ─────────────────────────────────────────────────────
    def fetch_api(self, endpoint: str, params: dict = None) -> pd.DataFrame:
        print(f"  [fetch_api] Запрос к {endpoint}...")
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df["source"] = f"api:{endpoint}"
        df["collected_at"] = datetime.now().isoformat()
        return df

    # ── skill: merge ─────────────────────────────────────────────────────────
    def merge(self, sources: list) -> pd.DataFrame:
        print(f"  [merge] Объединяю {len(sources)} источников...")
        df = pd.concat(sources, ignore_index=True)
        df = df[["text", "label", "source", "collected_at"]]
        df = df.dropna(subset=["text", "label"])
        df = df.reset_index(drop=True)
        print(f"  [merge] Итого {len(df)} записей")
        return df

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self, sources: list = None) -> pd.DataFrame:
        if sources is None:
            sources = self.config.get("sources", [])

        print(f"\n[DataCollectionAgent] Задача: '{self.task}'")
        print(f"[DataCollectionAgent] Источников: {len(sources)}\n")

        dataframes = []
        for src in sources:
            src_type = src.get("type")

            if src_type == "hf_dataset":
                df = self.load_dataset(
                    name=src["name"],
                    source="hf",
                    split=src.get("split", "train"),
                    sample=src.get("sample"),
                    label_map=src.get("label_map"),
                    text_col=src.get("text_col"),
                    config_name=src.get("config_name")
                )
            elif src_type == "scrape":
                df = self.scrape(
                    url=src["url"],
                    selector=src["selector"],
                    pages=src.get("pages", 1),
                    text_selector=src.get("text_selector"),
                    fixed_label=src.get("fixed_label"),
                    label_from_class=src.get("label_from_class"),
                    label_map=src.get("label_map"),
                    page_param=src.get("page_param"),
                )
            elif src_type == "api":
                df = self.fetch_api(
                    endpoint=src["endpoint"],
                    params=src.get("params", {})
                )
            else:
                print(f"  [run] Неизвестный тип источника: {src_type}, пропускаю")
                continue

            dataframes.append(df)

        if not dataframes:
            raise ValueError("Нет данных — все источники вернули пустые результаты")

        result = self.merge(dataframes)

        # Сохраняем
        os.makedirs(os.path.join(_ROOT, "data", "raw"), exist_ok=True)
        result.to_parquet(os.path.join(_ROOT, "data", "raw", "collected.parquet"), index=False)
        result.to_csv(os.path.join(_ROOT, "data", "raw", "collected.csv"), index=False)
        print(f"\n[DataCollectionAgent] Данные сохранены в data/raw/")
        print(f"[DataCollectionAgent] Финальный датасет: {result.shape}")
        print(result["label"].value_counts().to_string())

        return result


if __name__ == "__main__":
    agent = DataCollectionAgent(config=os.path.join(_ROOT, "config.yaml"))
    df = agent.run()
    print("\nПервые 3 записи:")
    print(df.head(3).to_string())
