"""
DataQualityAgent — Задание 2
Агент-детектив: выявляет и устраняет проблемы качества данных.
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class DataQualityAgent:

    # ── skill: detect_issues ─────────────────────────────────────────────────
    def detect_issues(self, df: pd.DataFrame) -> dict:
        print("[DataQualityAgent] Анализирую данные...")

        # 1. Пропущенные значения
        missing = df.isnull().sum().to_dict()
        missing_pct = (df.isnull().mean() * 100).round(2).to_dict()

        # 2. Дубликаты
        duplicates = int(df.duplicated().sum())

        # 3. Выбросы по длине текста (IQR)
        df["_text_len"] = df["text"].str.len()
        Q1 = df["_text_len"].quantile(0.25)
        Q3 = df["_text_len"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (df["_text_len"] < lower) | (df["_text_len"] > upper)
        outliers = df[outlier_mask].index.tolist()
        df.drop(columns=["_text_len"], inplace=True)

        # 4. Дисбаланс классов
        label_counts = df["label"].value_counts().to_dict()
        counts = list(label_counts.values())
        imbalance_ratio = round(max(counts) / min(counts), 2) if min(counts) > 0 else None

        report = {
            "missing": {"counts": missing, "percent": missing_pct},
            "duplicates": duplicates,
            "outliers": {"count": len(outliers), "indices": outliers[:10], "bounds": {"lower": round(lower, 1), "upper": round(upper, 1)}},
            "imbalance": {"label_counts": label_counts, "ratio": imbalance_ratio}
        }

        print(f"  Пропуски: {sum(v for v in missing.values() if v > 0)} ячеек")
        print(f"  Дубликаты: {duplicates}")
        print(f"  Выбросы по длине текста: {len(outliers)}")
        print(f"  Дисбаланс классов: {imbalance_ratio}x")

        return report

    # ── skill: fix ───────────────────────────────────────────────────────────
    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        print(f"\n[DataQualityAgent] Применяю стратегию: {strategy}")
        df = df.copy()

        # Дубликаты
        dup_strategy = strategy.get("duplicates", "drop")
        if dup_strategy == "drop":
            before = len(df)
            df = df.drop_duplicates(subset=["text"])
            print(f"  Дубликаты удалены: {before - len(df)} строк")

        # Пропущенные значения
        missing_strategy = strategy.get("missing", "drop")
        if missing_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=["text", "label"])
            print(f"  Пропуски удалены: {before - len(df)} строк")
        elif missing_strategy == "fill":
            df["text"] = df["text"].fillna("")
            df["label"] = df["label"].fillna("unknown")
            print(f"  Пропуски заполнены пустыми значениями")

        # Выбросы
        outlier_strategy = strategy.get("outliers", "clip_iqr")
        df["_text_len"] = df["text"].str.len()
        Q1 = df["_text_len"].quantile(0.25)
        Q3 = df["_text_len"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if outlier_strategy == "clip_iqr":
            before = len(df)
            df = df[(df["_text_len"] >= lower) & (df["_text_len"] <= upper)]
            print(f"  Выбросы удалены (IQR): {before - len(df)} строк")
        elif outlier_strategy == "keep":
            print(f"  Выбросы оставлены без изменений")

        df = df.drop(columns=["_text_len"]).reset_index(drop=True)
        print(f"  Итого после чистки: {len(df)} записей")
        return df

    # ── skill: compare ───────────────────────────────────────────────────────
    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        print("\n[DataQualityAgent] Сравниваю до/после...")

        def stats(df):
            return {
                "Записей": len(df),
                "Пропусков": int(df.isnull().sum().sum()),
                "Дубликатов": int(df.duplicated().sum()),
                "Выбросов (IQR)": int(self._count_outliers(df)),
                "Баланс (ratio)": round(df["label"].value_counts().max() / df["label"].value_counts().min(), 2),
                "Средняя длина текста": round(df["text"].str.len().mean(), 1),
            }

        before = stats(df_before)
        after = stats(df_after)

        comparison = pd.DataFrame({
            "Метрика": list(before.keys()),
            "До": list(before.values()),
            "После": list(after.values()),
        })
        comparison["Изменение"] = comparison.apply(
            lambda r: f"{r['После'] - r['До']:+.1f}" if isinstance(r["До"], (int, float)) else "-",
            axis=1
        )
        return comparison

    def _count_outliers(self, df):
        lengths = df["text"].str.len()
        Q1, Q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        IQR = Q3 - Q1
        return ((lengths < Q1 - 1.5 * IQR) | (lengths > Q3 + 1.5 * IQR)).sum()

    # ── bonus: llm_explain ───────────────────────────────────────────────────
    def llm_explain(self, report: dict, task_description: str) -> str:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "ANTHROPIC_API_KEY не найден в .env"

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Ты эксперт по качеству данных для ML.

Задача: {task_description}

Отчёт о проблемах в датасете:
- Пропущенные значения: {report['missing']['counts']}
- Дубликаты: {report['duplicates']}
- Выбросы по длине текста: {report['outliers']['count']} (границы: {report['outliers']['bounds']})
- Дисбаланс классов: {report['imbalance']}

Объясни найденные проблемы простыми словами и порекомендуй стратегию чистки. Отвечай кратко, по пунктам."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(_ROOT, "data", "raw", "collected.parquet"))
    print(f"Загружено {len(df)} записей\n")

    agent = DataQualityAgent()

    # Detect
    report = agent.detect_issues(df)

    # Fix — стратегия 1
    df_clean = agent.fix(df, strategy={
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "clip_iqr"
    })

    # Compare
    comparison = agent.compare(df, df_clean)
    print("\nСравнение до/после:")
    print(comparison.to_string(index=False))

    # LLM бонус
    print("\n[LLM] Объяснение проблем:")
    explanation = agent.llm_explain(report, "бинарная классификация тональности отзывов (positive/negative)")
    print(explanation)

    # Сохраняем
    df_clean.to_parquet(os.path.join(_ROOT, "data", "raw", "collected_clean.parquet"), index=False)
    print("\nСохранено: data/raw/collected_clean.parquet")
