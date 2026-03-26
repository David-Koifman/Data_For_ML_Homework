# Skill: Annotation

## Цель
Автоматически разметить данные, сгенерировать спецификацию разметки, оценить качество и экспортировать в LabelStudio.

## Когда спрашивать пользователя
- **Спроси объём** — сколько примеров разметить (по умолчанию весь датасет, но можно меньше для экономии API)
- **Спроси порог уверенности** — по умолчанию 0.7, можно изменить
- НЕ спрашивай про технические детали разметки

## Порядок действий

1. **Загрузить данные** — из `data/raw/collected_clean.parquet`

2. **Спросить объём** — предложи варианты:
   ```
   Датасет содержит 1867 записей.
   Разметить:
   A) Весь датасет (займёт ~5 мин, потратит ~$0.10 API)
   B) Выборку 200 записей (быстро, для теста)
   Выбери (A/B):
   ```

3. **Авторазметка** — запусти скрипт:
   ```
   skills/annotation/scripts/annotation_agent.py
   ```
   Метод `auto_label(df, modality='text')` → DataFrame с колонками `auto_label`, `confidence`

4. **Сгенерировать спецификацию** — метод `generate_spec(df, task)` → `annotation_spec.md`

5. **Проверить качество** — метод `check_quality(df_labeled)`:
   - Cohen's κ (согласие с оригинальными метками)
   - Распределение меток
   - Средняя уверенность

6. **Флагировать неуверенные** — примеры с `confidence < threshold` → `review_queue.csv`

7. **Экспорт в LabelStudio** — метод `export_to_labelstudio(df)` → `labelstudio_import.json`

8. **Сохранить** — размеченный датасет в `data/labeled/collected_labeled.parquet`

9. **Сообщить итог**:
   - Сколько размечено
   - Cohen's κ
   - Сколько флагов для проверки
   - Путь к `annotation_spec.md` и `labelstudio_import.json`

## Выходной формат
```python
pd.DataFrame с колонками: text, label, auto_label, confidence
Сохранено в: data/labeled/collected_labeled.parquet
Спецификация: annotation_spec.md
LabelStudio: labelstudio_import.json
Флаги: review_queue.csv
```

## Доступные скрипты
- `scripts/annotation_agent.py` — агент (скиллы: auto_label, generate_spec, check_quality, export_to_labelstudio)
