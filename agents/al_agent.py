"""
ActiveLearningAgent — Задание 4 (Трек A)
Умный отбор данных для разметки.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

load_dotenv()


class ActiveLearningAgent:
    def __init__(self, model: str = "logreg"):
        self.model_type = model
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None

    # ── skill: fit ───────────────────────────────────────────────────────────
    def fit(self, labeled_df: pd.DataFrame):
        X = self.vectorizer.fit_transform(labeled_df["text"])
        y = labeled_df["label"].values
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, y)
        return self.model

    # ── skill: query ─────────────────────────────────────────────────────────
    def query(self, pool: pd.DataFrame, strategy: str = "entropy", batch_size: int = 20) -> list:
        if strategy == "random":
            return pool.sample(n=min(batch_size, len(pool)), random_state=None).index.tolist()

        X_pool = self.vectorizer.transform(pool["text"])
        proba = self.model.predict_proba(X_pool)

        if strategy == "entropy":
            # Высокая энтропия = модель не уверена
            scores = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        elif strategy == "margin":
            # Маленький margin = два класса почти одинаково вероятны
            sorted_proba = np.sort(proba, axis=1)[:, ::-1]
            scores = 1 - (sorted_proba[:, 0] - sorted_proba[:, 1])
        else:
            raise ValueError(f"Стратегия '{strategy}' не поддерживается. Используй entropy/margin/random")

        # Берём топ-N с наивысшим score
        top_indices = np.argsort(scores)[::-1][:batch_size]
        return pool.iloc[top_indices].index.tolist()

    # ── skill: evaluate ──────────────────────────────────────────────────────
    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        X_test = self.vectorizer.transform(test_df["text"])
        y_test = test_df["label"].values
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred, pos_label="positive", average="binary"), 4),
            "n_labeled": len(labeled_df)
        }

    # ── skill: report ────────────────────────────────────────────────────────
    def report(self, history_entropy: list, history_random: list = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_entropy = [h["n_labeled"] for h in history_entropy]
        acc_entropy = [h["accuracy"] for h in history_entropy]
        f1_entropy = [h["f1"] for h in history_entropy]

        axes[0].plot(n_entropy, acc_entropy, "o-", color="#2ecc71", linewidth=2, label="Entropy")
        axes[1].plot(n_entropy, f1_entropy, "o-", color="#2ecc71", linewidth=2, label="Entropy")

        if history_random:
            n_random = [h["n_labeled"] for h in history_random]
            acc_random = [h["accuracy"] for h in history_random]
            f1_random = [h["f1"] for h in history_random]
            axes[0].plot(n_random, acc_random, "s--", color="#e74c3c", linewidth=2, label="Random")
            axes[1].plot(n_random, f1_random, "s--", color="#e74c3c", linewidth=2, label="Random")

        axes[0].set_title("Кривая обучения — Accuracy")
        axes[0].set_xlabel("Количество размеченных примеров")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Кривая обучения — F1")
        axes[1].set_xlabel("Количество размеченных примеров")
        axes[1].set_ylabel("F1 Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Active Learning: Entropy vs Random", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
        plt.savefig(os.path.join(_ROOT, "reports", "learning_curve.png"), dpi=150)
        plt.close()
        print("Сохранено: reports/learning_curve.png")

    # ── run_cycle ────────────────────────────────────────────────────────────
    def run_cycle(self, labeled_df: pd.DataFrame, pool_df: pd.DataFrame,
                  strategy: str = "entropy", n_iterations: int = 5,
                  batch_size: int = 20, test_df: pd.DataFrame = None) -> list:

        print(f"[ALAgent] Стратегия: {strategy} | Старт: {len(labeled_df)} | Итераций: {n_iterations} | Батч: {batch_size}")

        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()
        history = []

        for i in range(n_iterations + 1):
            # Обучаем
            self.fit(current_labeled)

            # Оцениваем
            metrics = self.evaluate(current_labeled, test_df)
            metrics["iteration"] = i
            history.append(metrics)

            print(f"  Итерация {i}: n={metrics['n_labeled']}, acc={metrics['accuracy']}, f1={metrics['f1']}")

            if i < n_iterations and len(current_pool) >= batch_size:
                # Выбираем следующие примеры
                indices = self.query(current_pool, strategy=strategy, batch_size=batch_size)
                new_examples = current_pool.loc[indices]
                current_labeled = pd.concat([current_labeled, new_examples], ignore_index=True)
                current_pool = current_pool.drop(indices)

        print(f"  Финальная модель: {len(current_labeled)} примеров")
        return history

    # ── bonus: llm_analyze ───────────────────────────────────────────────────
    def llm_analyze(self, history_entropy: list, history_random: list) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        final_entropy = history_entropy[-1]
        final_random = history_random[-1]

        prompt = f"""Проанализируй результаты Active Learning эксперимента:

Стратегия Entropy (умный отбор):
{history_entropy}

Стратегия Random (случайный отбор):
{history_random}

Финальные метрики:
- Entropy: accuracy={final_entropy['accuracy']}, f1={final_entropy['f1']}, n={final_entropy['n_labeled']}
- Random:  accuracy={final_random['accuracy']}, f1={final_random['f1']}, n={final_random['n_labeled']}

Ответь на вопросы:
1. Какая стратегия лучше и почему?
2. Сколько примеров сэкономила entropy vs random при том же качестве?
3. Практический вывод для задачи sentiment analysis.

Отвечай кратко, по пунктам, на русском."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(_ROOT, "data", "labeled", "collected_labeled.parquet"))
    df = df[df["auto_label"].isin(["positive", "negative"])].copy()
    df["label"] = df["auto_label"]
    print(f"Загружено {len(df)} записей\n")

    # Разделяем на train pool и test
    test_df = df.sample(n=300, random_state=42)
    train_df = df.drop(test_df.index)

    # Стартовые 50 размеченных
    labeled_50 = train_df.sample(n=50, random_state=42)
    pool_df = train_df.drop(labeled_50.index)

    agent = ActiveLearningAgent(model="logreg")

    # Entropy
    print("=== Стратегия: ENTROPY ===")
    history_entropy = agent.run_cycle(
        labeled_df=labeled_50.copy(),
        pool_df=pool_df.copy(),
        strategy="entropy",
        n_iterations=5,
        batch_size=20,
        test_df=test_df
    )

    # Random
    print("\n=== Стратегия: RANDOM ===")
    history_random = agent.run_cycle(
        labeled_df=labeled_50.copy(),
        pool_df=pool_df.copy(),
        strategy="random",
        n_iterations=5,
        batch_size=20,
        test_df=test_df
    )

    # Report
    agent.report(history_entropy, history_random)

    # LLM анализ
    print("\n[LLM] Анализ результатов:")
    print(agent.llm_analyze(history_entropy, history_random))

    # Сохраняем финальную модель
    import joblib
    os.makedirs(os.path.join(_ROOT, "models"), exist_ok=True)
    joblib.dump(agent.model, os.path.join(_ROOT, "models", "final_model.pkl"))
    joblib.dump(agent.vectorizer, os.path.join(_ROOT, "models", "vectorizer.pkl"))
    print("\nМодель сохранена: models/final_model.pkl")
