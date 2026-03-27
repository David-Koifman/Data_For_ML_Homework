# Data Pipeline Project — ML Annotation Pipeline

## Идея

Проект решает задачу: **как разметить большой датасет дёшево и быстро**.

Разметка вручную — дорого и долго. Использовать LLM на весь датасет — дорого. Решение — пайплайн из 4 агентов:

```
Собрать данные → Почистить → Claude размечает ВЫБОРКУ → Модель размечает ВСЕХ
```

Claude смотрит на 20% данных, обученная модель размечает оставшиеся 80% бесплатно. Human-in-the-Loop проверяет только сомнительные случаи.

---

## Технологии

| Что | Зачем |
|-----|-------|
| Claude API (haiku) | Авторазметка выборки, объяснение проблем, анализ AL |
| HuggingFace datasets | Готовые датасеты для обучения |
| BeautifulSoup + requests | Scraping второго источника данных |
| TF-IDF + LogisticRegression | Быстрая интерпретируемая модель для финальной разметки |
| Active Learning (entropy) | Умный отбор данных — меньше разметки, то же качество |
| Streamlit | UI для HITL-проверки |
| Claude Code Skills | Оркестрация всего пайплайна через агентов |

---

## Архитектура — 4 агента

### DataCollectionAgent — Задание 1
Собирает данные из двух источников и объединяет в единый DataFrame.
- Источник 1: HuggingFace dataset (готовый датасет)
- Источник 2: Web scraping (BeautifulSoup)
- Конфигурация через `config.yaml` — тему и источники можно менять без изменения кода
- Выход: `data/raw/collected.parquet` — колонки `[text, label, source, collected_at]`

### DataQualityAgent — Задание 2
Находит и устраняет проблемы в данных.
- Обнаруживает: пропуски, дубликаты, выбросы по длине текста (IQR), дисбаланс классов
- Применяет стратегию чистки и сравнивает до/после
- Бонус: Claude объясняет найденные проблемы и рекомендует стратегию
- Выход: `data/raw/collected_clean.parquet`

### AnnotationAgent — Задание 3
Размечает выборку данных через Claude API (zero-shot classification).
- Берёт 20% датасета (max 1000 записей) — не весь, это ключевой момент
- Отправляет батчами по 20 записей в claude-haiku
- Возвращает `auto_label` + `confidence` для каждого текста
- Примеры с confidence < 0.7 → `review_queue.csv` на HITL-проверку
- Генерирует спецификацию разметки (`annotation_spec.md`)
- Экспортирует в LabelStudio формат
- Выход: `data/labeled/collected_labeled.parquet`

### ActiveLearningAgent — Задание 4
Обучает модель на размеченной выборке, сравнивает стратегии отбора данных.
- Модель: LogisticRegression + TF-IDF
- Стратегии: Entropy (умный отбор) vs Random (случайный)
- Цикл: 50 стартовых примеров → 5 итераций по 20 → итого 150 размеченных
- Entropy выбирает примеры где модель наиболее не уверена → учится быстрее
- Финальная модель размечает весь датасет
- Бонус: Claude анализирует результаты эксперимента
- Выход: `models/final_model.pkl`, `reports/learning_curve.png`

---

## Алгоритм пайплайна

```
config.yaml (тема + источники)
    ↓
Шаг 1: DataCollectionAgent
        HuggingFace + scraping → data/raw/collected.parquet
    ↓
Шаг 2: DataQualityAgent
        дубликаты + выбросы + пропуски → data/raw/collected_clean.parquet
    ↓
Шаг 3: AnnotationAgent
        Claude размечает 20% выборку → data/labeled/collected_labeled.parquet
        низкая уверенность → review_queue.csv
    ↓
Шаг 4: HITL
        человек проверяет review_queue.csv → review_queue_corrected.csv
    ↓
Шаг 5: ActiveLearningAgent
        учится на размеченной выборке (entropy vs random)
        → models/final_model.pkl
    ↓
Шаг 6: Финальная разметка
        модель размечает ВЕСЬ датасет → data/labeled/final_dataset.parquet
```

---

## Пример: классификация спама

**Задача:** определить является ли SMS-сообщение спамом.

**Данные:**
| Источник | Записей |
|----------|---------|
| HuggingFace `sms_spam` | 1500 |
| Scraping `quotes.toscrape.com` (ham примеры) | 100 |
| После чистки (дубликаты + выбросы) | **1458** |

**Разметка:**
- Claude разметил 300 записей (21% выборка)
- Уверенность: 0.814 — хорошо, спам легко распознаётся
- Флагов для HITL: 36 (Singlish и сокращённые тексты)
- HITL результат: все 36 подтверждены как ham (Claude не ошибся)

**Active Learning:**
| Итерация | Примеров | Entropy F1 | Random F1 |
|----------|----------|------------|-----------|
| 0 | 50 | 0.00 | 0.00 |
| 1 | 70 | 0.25 | 0.00 |
| **2** | **90** | **0.80** | **0.00** |
| 3 | 110 | 0.44 | 0.00 |
| 5 | 150 | 0.00 | 0.00 |

Entropy на 90 примерах достигла F1=0.80 — нашла спам-примеры умным отбором. Random так и не нашёл спам из-за дисбаланса (спама всего 11% в датасете).

**Финальный датасет:**
```
data/labeled/final_dataset.csv  — 1458 записей
spam (positive):   164  (11%)
ham  (negative):  1294  (89%)
Уверенность модели: 0.672
```

---

## Запуск

```bash
pip install -r requirements.txt

# Полный пайплайн
python run_pipeline.py

# Отдельные агенты
python agents/data_collection_agent.py
python agents/data_quality_agent.py
python agents/annotation_agent.py
python agents/al_agent.py

# HITL дашборд
streamlit run app.py
```

---

## Ключевые файлы

```
config.yaml                         ← тема и источники данных (меняй здесь)
data/raw/collected.parquet          ← шаг 1: сырые данные
data/raw/collected_clean.parquet    ← шаг 2: после чистки
data/labeled/collected_labeled.parquet  ← шаг 3: метки Claude (выборка)
review_queue.csv                    ← шаг 3: на HITL-проверку
data/labeled/final_dataset.parquet  ← шаг 6: финал, весь датасет
models/final_model.pkl              ← обученная модель
reports/learning_curve.png          ← график entropy vs random
reports/al_report.md                ← метрики AL по итерациям
annotation_spec.md                  ← спецификация разметки
```

---

## Бонусы

- **+3 балла:** Claude API в 3 агентах (annotation, quality explain, AL analyze)
- **+2 балла:** Streamlit дашборд (`streamlit run app.py`) — HITL + метрики + датасет
