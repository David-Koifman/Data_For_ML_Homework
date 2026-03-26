"""
Финальный пайплайн — Data Project
Запуск: python run_pipeline.py

Шаги:
1. Сбор данных      — DataCollectionAgent
2. Чистка данных    — DataQualityAgent
3. Авторазметка     — AnnotationAgent
4. ❗ HITL          — человек проверяет review_queue.csv
5. AL отбор         — ActiveLearningAgent
6. Обучение модели  — финальная модель
7. Отчёт            — метрики всех этапов
"""

import sys
import os
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent

BANNER = "=" * 55

def log(step, msg):
    print(f"\n{BANNER}")
    print(f"  ШАГ {step}: {msg}")
    print(BANNER)


# ── Шаг 1: Сбор данных ──────────────────────────────────────
def step1_collect():
    log(1, "Сбор данных — DataCollectionAgent")
    agent = DataCollectionAgent(config="config.yaml")
    df = agent.run()
    print(f"  Собрано: {len(df)} записей")
    return df


# ── Шаг 2: Чистка данных ────────────────────────────────────
def step2_clean(df):
    log(2, "Чистка данных — DataQualityAgent")
    agent = DataQualityAgent()

    report = agent.detect_issues(df)

    df_clean = agent.fix(df, strategy={
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "clip_iqr"
    })

    comparison = agent.compare(df, df_clean)

    # Сохраняем отчёт
    with open("reports/quality_report.md", "w") as f:
        f.write("# Quality Report\n\n")
        f.write(f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(f"## Исходный датасет\n- Записей: {len(df)}\n\n")
        f.write(f"## Найденные проблемы\n")
        f.write(f"- Пропуски: {sum(report['missing']['counts'].values())}\n")
        f.write(f"- Дубликаты: {report['duplicates']}\n")
        f.write(f"- Выбросы: {report['outliers']['count']}\n")
        f.write(f"- Дисбаланс классов: {report['imbalance']['ratio']}x\n\n")
        f.write(f"## После чистки\n- Записей: {len(df_clean)}\n\n")
        f.write(f"## Сравнение до/после\n```\n{comparison.to_string(index=False)}\n```\n")

    print(f"  После чистки: {len(df_clean)} записей")
    print(f"  Отчёт: reports/quality_report.md")
    return df_clean


# ── Шаг 3: Авторазметка ─────────────────────────────────────
def step3_annotate(df):
    log(3, "Авторазметка — AnnotationAgent")

    # Если данные уже размечены — используем их
    labeled_path = "data/labeled/collected_labeled.parquet"
    if os.path.exists(labeled_path):
        df_labeled = pd.read_parquet(labeled_path)
        print(f"  Используем уже размеченные данные: {len(df_labeled)} записей")
        print(f"  (чтобы перезапустить разметку — удали {labeled_path})")
    else:
        agent = AnnotationAgent(modality="text", confidence_threshold=0.7)
        df_labeled = agent.auto_label(df)
        agent.generate_spec(df_labeled, task="sentiment_classification")
        df_labeled.to_parquet(labeled_path, index=False)

    metrics = {
        "total": len(df_labeled),
        "label_dist": df_labeled["auto_label"].value_counts().to_dict(),
        "confidence_mean": round(float(df_labeled["confidence"].mean()), 3),
        "low_conf": int((df_labeled["confidence"] < 0.7).sum())
    }

    # Сохраняем отчёт
    with open("reports/annotation_report.md", "w") as f:
        f.write("# Annotation Report\n\n")
        f.write(f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(f"## Результаты авторазметки\n")
        f.write(f"- Всего записей: {metrics['total']}\n")
        f.write(f"- Распределение: {metrics['label_dist']}\n")
        f.write(f"- Средняя уверенность: {metrics['confidence_mean']}\n")
        f.write(f"- Флагов для проверки: {metrics['low_conf']}\n")

    print(f"  Размечено: {metrics['total']} записей")
    print(f"  Уверенность: {metrics['confidence_mean']}")
    print(f"  Флагов: {metrics['low_conf']} → review_queue.csv")
    return df_labeled


# ── Шаг 4: HITL ─────────────────────────────────────────────
def step4_human_review(df_labeled):
    log(4, "❗ HUMAN-IN-THE-LOOP")

    low_conf = df_labeled[df_labeled["confidence"] < 0.7].copy()
    low_conf.to_csv("review_queue.csv", index=False)

    print(f"\n  Примеров с низкой уверенностью: {len(low_conf)}")
    print(f"  Файл для проверки: review_queue.csv")
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  ДЕЙСТВИЕ: Открой review_queue.csv          │")
    print(f"  │  Проверь колонку auto_label                 │")
    print(f"  │  Исправь ошибки в колонке auto_label        │")
    print(f"  │  Сохрани как review_queue_corrected.csv     │")
    print(f"  └─────────────────────────────────────────────┘")
    input("\n  Нажми Enter когда закончишь проверку...")

    corrected_path = "review_queue_corrected.csv"
    if os.path.exists(corrected_path):
        corrected = pd.read_csv(corrected_path)
        high_conf = df_labeled[df_labeled["confidence"] >= 0.7]
        df_reviewed = pd.concat([high_conf, corrected], ignore_index=True)
        print(f"\n  Принято исправлений: {len(corrected)}")
    else:
        print(f"\n  review_queue_corrected.csv не найден — используем оригинальные метки")
        df_reviewed = df_labeled.copy()

    df_reviewed["label"] = df_reviewed["auto_label"]
    df_reviewed = df_reviewed[df_reviewed["label"].isin(["positive", "negative"])]
    df_reviewed.to_parquet("data/labeled/reviewed.parquet", index=False)
    print(f"  Итого после проверки: {len(df_reviewed)} записей")
    return df_reviewed


# ── Шаг 5: AL отбор + обучение ──────────────────────────────
def step5_train(df_reviewed):
    log(5, "Active Learning + Обучение — ALAgent")

    agent = ActiveLearningAgent(model="logreg")

    test_df = df_reviewed.sample(n=min(300, len(df_reviewed)//5), random_state=42)
    train_df = df_reviewed.drop(test_df.index)
    labeled_50 = train_df.sample(n=min(50, len(train_df)//3), random_state=42)
    pool_df = train_df.drop(labeled_50.index)

    history = agent.run_cycle(
        labeled_df=labeled_50,
        pool_df=pool_df,
        strategy="entropy",
        n_iterations=5,
        batch_size=20,
        test_df=test_df
    )

    agent.report(history)

    final_metrics = history[-1]

    # Сохраняем отчёт
    with open("reports/al_report.md", "w") as f:
        f.write("# Active Learning Report\n\n")
        f.write(f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(f"## Конфигурация\n")
        f.write(f"- Модель: LogisticRegression + TF-IDF\n")
        f.write(f"- Стратегия: Entropy\n")
        f.write(f"- Старт: 50 примеров, 5 итераций по 20\n\n")
        f.write(f"## История обучения\n")
        for h in history:
            f.write(f"- Итерация {h['iteration']}: n={h['n_labeled']}, acc={h['accuracy']}, f1={h['f1']}\n")
        f.write(f"\n## Финальные метрики\n")
        f.write(f"- Accuracy: {final_metrics['accuracy']}\n")
        f.write(f"- F1: {final_metrics['f1']}\n")

    # Сохраняем модель
    joblib.dump(agent.model, "models/final_model.pkl")
    joblib.dump(agent.vectorizer, "models/vectorizer.pkl")

    print(f"  Финальные метрики: accuracy={final_metrics['accuracy']}, F1={final_metrics['f1']}")
    print(f"  Модель сохранена: models/final_model.pkl")
    return final_metrics, history


# ── Шаг 6: Итоговый отчёт ───────────────────────────────────
def step6_report(metrics_per_step):
    log(6, "Итоговый отчёт")

    print("\n" + BANNER)
    print("  ИТОГИ ПАЙПЛАЙНА")
    print(BANNER)
    for step, info in metrics_per_step.items():
        print(f"  {step}: {info}")

    print(f"\n  Все отчёты в папке reports/")
    print(f"  Кривая обучения: reports/learning_curve.png")
    print(f"  Финальная модель: models/final_model.pkl")


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BANNER)
    print("  DATA PIPELINE — старт")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(BANNER)

    metrics = {}

    # Шаг 1
    df_raw = step1_collect()
    metrics["Шаг 1 (Сбор)"] = f"{len(df_raw)} записей"

    # Шаг 2
    df_clean = step2_clean(df_raw)
    metrics["Шаг 2 (Чистка)"] = f"{len(df_clean)} записей"

    # Шаг 3
    df_labeled = step3_annotate(df_clean)
    metrics["Шаг 3 (Разметка)"] = f"{len(df_labeled)} записей, conf={df_labeled['confidence'].mean():.3f}"

    # Шаг 4
    df_reviewed = step4_human_review(df_labeled)
    metrics["Шаг 4 (HITL)"] = f"{len(df_reviewed)} записей после проверки"

    # Шаг 5
    final_metrics, history = step5_train(df_reviewed)
    metrics["Шаг 5 (Обучение)"] = f"accuracy={final_metrics['accuracy']}, F1={final_metrics['f1']}"

    # Шаг 6
    step6_report(metrics)

    print(f"\n{BANNER}")
    print("  ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО ✓")
    print(BANNER)
