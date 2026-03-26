# Задания курса — чеклист баллов

## Задание 1 — DataCollectionAgent (15 баллов) | Дедлайн: 13.03.2026
### Архитектура
- [ ] skill: scrape(url, selector) → DataFrame
- [ ] skill: fetch_api(endpoint, params) → DataFrame
- [ ] skill: load_dataset(name, source='hf'|'kaggle') → DataFrame
- [ ] skill: merge(sources: list[DataFrame]) → DataFrame
- [ ] agent.run(sources=[...]) принимает список источников из config.yaml

### Требования
- [ ] Минимум 2 источника (один HuggingFace/Kaggle, один scraping/API)
- [ ] Унифицированная схема: колонки text, label, source, collected_at
- [ ] EDA: распределение классов, длины текстов, топ-20 слов
- [ ] README.md с описанием задачи ML, схемы данных, инструкцией запуска
- [ ] requirements.txt или pyproject.toml

### Файлы
- [ ] agents/data_collection_agent.py
- [ ] config.yaml
- [ ] notebooks/eda.ipynb
- [ ] data/raw/

---

## Задание 2 — DataQualityAgent (15 баллов) | Дедлайн: 20.03.2026
### Архитектура
- [ ] skill: detect_issues(df) → QualityReport (missing, duplicates, outliers, imbalance)
- [ ] skill: fix(df, strategy: dict) → DataFrame
- [ ] skill: compare(df_before, df_after) → ComparisonReport

### Требования
- [ ] Минимум 3 типа проблем: пропуски, дубликаты, выбросы (IQR или z-score)
- [ ] Минимум 2 стратегии чистки в методе fix()
- [ ] Сравнительный отчёт до/после по каждой метрике
- [ ] Часть 1: Детектив — визуализация каждой проблемы
- [ ] Часть 2: Хирург — 2 стратегии, сравнительная таблица
- [ ] Часть 3: Аргумент — обоснование выбора стратегии (Markdown-ячейка)

### Бонус (+2 балла)
- [ ] LLM-скилл: Claude API объясняет проблемы и рекомендует стратегию

### Файлы
- [ ] agents/data_quality_agent.py
- [ ] notebooks/quality_analysis.ipynb

---

## Задание 3 — AnnotationAgent (15 баллов) | Дедлайн: 21.03.2026
### Архитектура
- [ ] skill: auto_label(df, modality) → DataFrame (text→zero-shot)
- [ ] skill: generate_spec(df, task) → AnnotationSpec (Markdown-файл)
- [ ] skill: check_quality(df_labeled) → QualityMetrics
- [ ] skill: export_to_labelstudio(df) → JSON

### Требования
- [ ] auto_label работает для текста (zero-shot классификация)
- [ ] Спецификация: задача, классы с определениями, 3+ примера на класс, граничные случаи
- [ ] Экспорт в LabelStudio JSON — загружается без ошибок
- [ ] Метрики качества: Cohen's κ + распределение меток
- [ ] Передать спецификацию однокурснику → сравнить его разметку с авторазметкой

### Бонус (+2 балла)
- [ ] Human-in-the-loop: флагать confidence < threshold → отдельный файл для ручной разметки

### Файлы
- [ ] agents/annotation_agent.py
- [ ] annotation_spec.md
- [ ] labelstudio_import.json

---

## Задание 4 — ALAgent / ТРЕК A (15 баллов) | Дедлайн: 27.03.2026
### Архитектура
- [ ] skill: fit(labeled_df) → model
- [ ] skill: query(pool, strategy) → indices (entropy, margin, random)
- [ ] skill: evaluate(labeled_df, test_df) → Metrics
- [ ] skill: report(history) → LearningCurve

### Требования
- [ ] AL-цикл: старт N=50 → 5 итераций по 20 примеров
- [ ] Сравнение стратегий: entropy vs random на одном графике
- [ ] Вывод: сколько примеров сэкономлено vs random baseline
- [ ] README + воспроизводимость

### Бонус (+1 балл)
- [ ] LLM-скилл (Claude API) в пайплайне

### Файлы
- [ ] agents/al_agent.py
- [ ] notebooks/al_experiment.ipynb

---

## Финальный пайплайн (40 баллов) | Дедлайн: 28.03.2026
### Критерии оценки
- [ ] Все 4 агента переиспользованы — 10 баллов
- [ ] Пайплайн запускается одной командой python run_pipeline.py — 10 баллов
- [ ] Реальная HITL-точка (не лог, а реальная правка данных) — 8 баллов
- [ ] Обученная модель + метрики accuracy/F1 — 7 баллов
- [ ] Финальный отчёт: все 5 разделов — 5 баллов

### Логика пайплайна
- [ ] Шаг 1: DataCollectionAgent — сбор из 2+ источников
- [ ] Шаг 2: DataQualityAgent — чистка данных
- [ ] Шаг 3: AnnotationAgent — авторазметка
- [ ] ❗ HITL: review_queue.csv → человек правит метки
- [ ] Шаг 4: ALAgent — отбор информативных примеров
- [ ] Шаг 5: Обучение модели
- [ ] Шаг 6: Отчёт

### Финальный отчёт (README) — 5 разделов
- [ ] 1. Описание задачи и датасета
- [ ] 2. Что делал каждый агент
- [ ] 3. Описание HITL-точки
- [ ] 4. Метрики на каждом этапе + итоговые метрики модели
- [ ] 5. Ретроспектива

### Бонусы
- [ ] +3 балла: LLM-агент (Claude API) в пайплайне
- [ ] +2 балла: дашборд Streamlit/Gradio для HITL

### Файлы
- [ ] run_pipeline.py
- [ ] review_queue.csv
- [ ] data/labeled/
- [ ] models/
- [ ] reports/quality_report.md
- [ ] reports/annotation_report.md
- [ ] reports/al_report.md
- [ ] README.md
