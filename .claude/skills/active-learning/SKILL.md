---
name: active-learning
description: Smart data selection using Active Learning (entropy/margin/random strategies). Trains LogisticRegression+TF-IDF, compares strategies, generates learning curves, uses Claude API to analyze results.
---

# Skill: ActiveLearning

## Цель
Умный отбор данных для обучения модели. Показать что Entropy-стратегия эффективнее случайного отбора.

## Когда спрашивать пользователя
- **Спроси стратегию** — предложи entropy/margin/random, объясни разницу
- **Спроси параметры** — стартовый размер (по умолчанию 50), итерации (5), батч (20)
- НЕ спрашивай про технические детали модели

## Порядок действий

1. **Загрузить данные** — из `data/labeled/collected_labeled.parquet`
   - Оставить только binary: positive/negative
   - Разбить на train/test (20% test)

2. **Спросить стратегию**:
   ```
   Доступные стратегии Active Learning:
   A) Entropy — выбирает примеры где модель наименее уверена (рекомендуется)
   B) Margin — выбирает примеры где два класса почти одинаково вероятны
   C) Random — случайный отбор (baseline)

   Рекомендую запустить Entropy vs Random для сравнения.
   Запустить сравнение? (да/нет):
   ```

3. **Запустить AL-цикл** — используй скрипт:
   ```
   agents/al_agent.py
   ```
   Метод `run_cycle(labeled_df, pool_df, strategy, n_iterations=5, batch_size=20, test_df)`

4. **Сравнить стратегии** — запустить entropy И random, построить кривые обучения на одном графике

5. **LLM анализ** — вызвать `llm_analyze(history_entropy, history_random)` — Claude даст выводы

6. **Сохранить модель** — `models/final_model.pkl` + `models/vectorizer.pkl`

7. **Сообщить итог**:
   - Финальные метрики (accuracy, F1)
   - Сколько примеров сэкономлено vs random
   - Путь к `reports/learning_curve.png`

## Выходной формат
```python
history: list[{iteration, n_labeled, accuracy, f1}]
Модель: models/final_model.pkl
График: reports/learning_curve.png
Отчёт: reports/al_report.md
```

## Доступные скрипты
- `agents/al_agent.py` — агент (скиллы: fit, query, evaluate, report, run_cycle, llm_analyze)
