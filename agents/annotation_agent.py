"""
AnnotationAgent — Задание 3
Автоматически размечает данные, генерирует спецификацию,
оценивает качество и экспортирует в LabelStudio.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AnnotationAgent:
    def __init__(self, modality: str = "text", confidence_threshold: float = 0.7):
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    # ── skill: auto_label ────────────────────────────────────────────────────
    def auto_label(self, df: pd.DataFrame, modality: str = None, task: str = None) -> pd.DataFrame:
        modality = modality or self.modality
        # Читаем задачу из config.yaml если не передана явно
        if task is None:
            try:
                import yaml
                with open(os.path.join(_ROOT, "config.yaml")) as f:
                    cfg = yaml.safe_load(f)
                task = cfg.get("task", "бинарная классификация текста (positive/negative)")
            except Exception:
                task = "бинарная классификация текста (positive/negative)"
        print(f"[AnnotationAgent] Авторазметка {len(df)} записей (modality={modality})...")
        print(f"[AnnotationAgent] Задача: {task}")

        if modality != "text":
            raise ValueError(f"Модальность '{modality}' не поддерживается. Используй 'text'.")

        client = self._get_client()
        labels, confidences = [], []

        # Батчевая разметка по 20 записей за раз
        batch_size = 20
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            texts = batch["text"].tolist()

            prompt = f"""Ты эксперт-разметчик данных для задачи машинного обучения.

ЗАДАЧА: {task}

КЛАССЫ:
- "positive" — текст явно относится к позитивному примеру задачи
- "negative" — текст явно относится к негативному примеру задачи

ПРАВИЛА УВЕРЕННОСТИ:
- 0.95–1.00: однозначно, никаких сомнений
- 0.80–0.94: скорее всего правильно, небольшие сомнения
- 0.65–0.79: есть сомнения, но выбор обоснован
- 0.50–0.64: очень неоднозначно, почти 50/50

ГРАНИЧНЫЕ СЛУЧАИ — ставь низкую уверенность (0.50–0.65) если:
- текст содержит признаки обоих классов
- текст слишком короткий или невнятный
- тема не связана с задачей напрямую
- смешанный тон (и хвалит, и критикует)

ФОРМАТ ОТВЕТА — строго JSON список, ровно {len(texts)} элементов, по одному на каждый текст:
[{{"label": "positive", "confidence": 0.95}}, {{"label": "negative", "confidence": 0.72}}, ...]

Никаких пояснений, никакого markdown, только JSON. Классы строго: "positive" или "negative".

ТЕКСТЫ ДЛЯ РАЗМЕТКИ:
""" + "\n---\n".join([f"[{j+1}] {t[:400]}" for j, t in enumerate(texts)])

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text.strip()
                # убираем markdown блок если есть
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                result = json.loads(raw.strip())
                for item in result:
                    labels.append(item.get("label", "unknown"))
                    confidences.append(float(item.get("confidence", 0.5)))
            except Exception as e:
                print(f"  Ошибка в батче {i//batch_size + 1}: {e}")
                for _ in texts:
                    labels.append("unknown")
                    confidences.append(0.0)

            if (i // batch_size + 1) % 5 == 0:
                print(f"  Размечено {min(i + batch_size, len(df))}/{len(df)}...")

        df = df.copy()
        df["auto_label"] = labels
        df["confidence"] = confidences

        # Бонус: флагаем неуверенные примеры
        low_conf = df[df["confidence"] < self.confidence_threshold]
        if len(low_conf) > 0:
            low_conf.to_csv(os.path.join(_ROOT, "review_queue.csv"), index=False)
            print(f"  Флагов для ручной проверки: {len(low_conf)} → review_queue.csv")

        print(f"  Готово. Распределение меток:")
        print(f"  {df['auto_label'].value_counts().to_dict()}")
        print(f"  Средняя уверенность: {df['confidence'].mean():.3f}")
        return df

    # ── skill: generate_spec ─────────────────────────────────────────────────
    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        print(f"[AnnotationAgent] Генерирую спецификацию разметки...")
        client = self._get_client()

        # Берём примеры из датасета
        pos_examples = df[df.get("auto_label", df.get("label", "")) == "positive"]["text"].head(3).tolist()
        neg_examples = df[df.get("auto_label", df.get("label", "")) == "negative"]["text"].head(3).tolist()

        prompt = f"""Создай спецификацию разметки для задачи: {task}

Используй эти реальные примеры из датасета:
POSITIVE примеры:
{chr(10).join([f"- {t[:200]}" for t in pos_examples])}

NEGATIVE примеры:
{chr(10).join([f"- {t[:200]}" for t in neg_examples])}

Спецификация должна содержать:
1. Описание задачи
2. Классы с чёткими определениями
3. 3+ примера на каждый класс
4. Граничные случаи (когда сложно определить класс)
5. Правила разметки

Пиши на русском языке, в формате Markdown."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        spec = response.content[0].text

        spec_path = os.path.join(_ROOT, "annotation_spec.md")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(f"# Спецификация разметки — {task}\n\n")
            f.write(f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write(spec)

        print(f"  Сохранено: annotation_spec.md")
        return spec

    # ── skill: check_quality ─────────────────────────────────────────────────
    def check_quality(self, df_labeled: pd.DataFrame) -> dict:
        print(f"[AnnotationAgent] Проверяю качество разметки...")

        label_dist = df_labeled["auto_label"].value_counts().to_dict()
        confidence_mean = round(float(df_labeled["confidence"].mean()), 3)
        confidence_std = round(float(df_labeled["confidence"].std()), 3)
        low_conf_count = int((df_labeled["confidence"] < self.confidence_threshold).sum())

        # Cohen's kappa: сравниваем auto_label с оригинальным label если есть
        kappa = None
        if "label" in df_labeled.columns and "auto_label" in df_labeled.columns:
            from sklearn.metrics import cohen_kappa_score
            valid = df_labeled[df_labeled["auto_label"].isin(["positive", "negative"])]
            if len(valid) > 0:
                kappa = round(cohen_kappa_score(valid["label"], valid["auto_label"]), 3)
                print(f"  Cohen's κ (auto vs original): {kappa}")

        metrics = {
            "kappa": kappa,
            "label_dist": label_dist,
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "low_confidence_count": low_conf_count,
            "total": len(df_labeled)
        }

        print(f"  Распределение: {label_dist}")
        print(f"  Уверенность: {confidence_mean} ± {confidence_std}")
        print(f"  Низкая уверенность (<{self.confidence_threshold}): {low_conf_count}")
        return metrics

    # ── skill: export_to_labelstudio ─────────────────────────────────────────
    def export_to_labelstudio(self, df: pd.DataFrame) -> str:
        print(f"[AnnotationAgent] Экспорт в LabelStudio ({len(df)} записей)...")

        tasks = []
        for idx, row in df.iterrows():
            task = {
                "id": int(idx),
                "data": {
                    "text": str(row["text"])[:1000]
                },
                "annotations": [{
                    "id": int(idx),
                    "result": [{
                        "id": f"result_{idx}",
                        "type": "choices",
                        "value": {
                            "choices": [str(row.get("auto_label", row.get("label", "unknown")))]
                        },
                        "from_name": "sentiment",
                        "to_name": "text"
                    }],
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": datetime.now().isoformat(),
                    "confidence": float(row.get("confidence", 1.0))
                }]
            }
            tasks.append(task)

        output_path = os.path.join(_ROOT, "labelstudio_import.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"  Сохранено: labelstudio_import.json ({len(tasks)} задач)")
        return output_path


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(_ROOT, "data", "raw", "collected_clean.parquet"))
    # Берём 100 записей для теста
    df_sample = df.copy()
    print(f"Загружено {len(df_sample)} записей\n")

    agent = AnnotationAgent(modality="text", confidence_threshold=0.7)

    # auto_label
    df_labeled = agent.auto_label(df_sample)

    # generate_spec
    agent.generate_spec(df_labeled, task="sentiment_classification")

    # check_quality
    metrics = agent.check_quality(df_labeled)
    print(f"\nМетрики: {metrics}")

    # export_to_labelstudio
    agent.export_to_labelstudio(df_labeled)

    # Сохраняем
    os.makedirs(os.path.join(_ROOT, "data", "labeled"), exist_ok=True)
    df_labeled.to_parquet(os.path.join(_ROOT, "data", "labeled", "collected_labeled.parquet"), index=False)
    df_labeled.to_csv(os.path.join(_ROOT, "data", "labeled", "collected_labeled.csv"), index=False)
    print(f"\nСохранено: data/labeled/collected_labeled.parquet + collected_labeled.csv")
