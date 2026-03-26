"""
Streamlit дашборд — HITL-разметка и визуализация результатов
Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

ROOT = Path(__file__).parent

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")
st.title("ML Data Pipeline — Dashboard")

# ── Навигация ─────────────────────────────────────────────────
page = st.sidebar.radio("Раздел", [
    "🏠 Главная",
    "✏️ HITL — Проверка меток",
    "📊 Результаты пайплайна",
    "📁 Датасет"
])

# ══════════════════════════════════════════════════════════════
# ГЛАВНАЯ
# ══════════════════════════════════════════════════════════════
if page == "🏠 Главная":
    st.header("Как использовать")

    st.info("**Вариант 1 — CLI (автоматический)**\n```\npython run_pipeline.py\n```\nПайплайн сам проходит все шаги. HITL-пауза: нужно вручную отредактировать `review_queue.csv`.")

    st.success("**Вариант 2 — Streamlit (этот дашборд)**\n```\nstreamlit run app.py\n```\nВизуальный интерфейс для проверки меток и просмотра результатов.")

    st.markdown("---")
    st.markdown("### Шаги пайплайна")
    steps = {
        "Шаг 1 — Сбор данных": "data/raw/collected.parquet",
        "Шаг 2 — Чистка": "data/raw/collected_clean.parquet",
        "Шаг 3 — Разметка": "data/labeled/collected_labeled.parquet",
        "HITL — Проверка": "review_queue.csv",
        "Шаг 4 — AL + Модель": "models/final_model.pkl",
    }
    for step, path in steps.items():
        exists = (ROOT / path).exists()
        st.markdown(f"{'✅' if exists else '⏳'} **{step}** — `{path}`")

# ══════════════════════════════════════════════════════════════
# HITL
# ══════════════════════════════════════════════════════════════
elif page == "✏️ HITL — Проверка меток":
    st.header("Проверка меток с низкой уверенностью")

    queue_path = ROOT / "review_queue.csv"
    corrected_path = ROOT / "review_queue_corrected.csv"

    if not queue_path.exists():
        st.warning("Файл `review_queue.csv` не найден. Сначала запусти пайплайн.")
        st.stop()

    df = pd.read_csv(queue_path)
    st.info(f"Всего примеров для проверки: **{len(df)}**")

    # Загружаем уже сохранённые правки если есть
    if corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
    else:
        corrected = df.copy()

    if "corrected_label" not in corrected.columns:
        corrected["corrected_label"] = corrected.get("auto_label", corrected.get("label", ""))

    # Навигация по примерам
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    idx = st.session_state.idx
    row = df.iloc[idx]

    st.markdown(f"### Пример {idx + 1} / {len(df)}")
    st.markdown(f"**Уверенность Claude:** `{row.get('confidence', '?'):.2f}`")
    st.markdown("**Текст:**")
    st.text_area("", value=str(row["text"]), height=150, disabled=True, key="text_display")

    auto = str(row.get("auto_label", row.get("label", "positive")))
    current = str(corrected.iloc[idx].get("corrected_label", auto))

    col1, col2 = st.columns(2)
    with col1:
        choice = st.radio(
            "Метка:",
            ["positive", "negative"],
            index=0 if current == "positive" else 1,
            key=f"radio_{idx}"
        )
    with col2:
        st.metric("Claude сказал", auto)
        if auto != choice:
            st.warning("⚠️ Ты изменил метку")
        else:
            st.success("✅ Совпадает с Claude")

    corrected.at[idx, "corrected_label"] = choice

    # Кнопки навигации
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Назад") and idx > 0:
            st.session_state.idx -= 1
            st.rerun()
    with c2:
        if st.button("Вперёд →") and idx < len(df) - 1:
            st.session_state.idx += 1
            st.rerun()
    with c3:
        if st.button("💾 Сохранить всё", type="primary"):
            save_df = corrected.copy()
            save_df["auto_label"] = save_df["corrected_label"]
            save_df.to_csv(corrected_path, index=False)
            st.success(f"Сохранено в `review_queue_corrected.csv`")

    # Прогресс
    st.markdown("---")
    changed = (corrected["corrected_label"] != corrected.get("auto_label", corrected.get("label", ""))).sum()
    st.progress((idx + 1) / len(df))
    st.caption(f"Просмотрено: {idx + 1}/{len(df)} | Исправлено: {changed}")

# ══════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТЫ
# ══════════════════════════════════════════════════════════════
elif page == "📊 Результаты пайплайна":
    st.header("Метрики пайплайна")

    col1, col2, col3, col4 = st.columns(4)

    # Шаг 1
    raw_path = ROOT / "data/raw/collected.parquet"
    if raw_path.exists():
        n_raw = len(pd.read_parquet(raw_path))
        col1.metric("Собрано записей", n_raw)

    # Шаг 2
    clean_path = ROOT / "data/raw/collected_clean.parquet"
    if clean_path.exists():
        n_clean = len(pd.read_parquet(clean_path))
        col2.metric("После чистки", n_clean, delta=f"{n_clean - n_raw if raw_path.exists() else ''}")

    # Шаг 3
    labeled_path = ROOT / "data/labeled/collected_labeled.parquet"
    if labeled_path.exists():
        df_lab = pd.read_parquet(labeled_path)
        conf_mean = df_lab["confidence"].mean() if "confidence" in df_lab.columns else 0
        col3.metric("Средняя уверенность", f"{conf_mean:.3f}")

    # AL
    al_report = ROOT / "reports/al_report.md"
    if al_report.exists():
        content = al_report.read_text()
        for line in content.split("\n"):
            if "accuracy" in line.lower():
                import re
                m = re.search(r"accuracy[:\s=]+([0-9.]+)", line, re.I)
                if m:
                    col4.metric("Финальная Accuracy", m.group(1))
                    break

    st.markdown("---")

    # Графики
    curve_path = ROOT / "reports/learning_curve.png"
    quality_path = ROOT / "reports/quality_issues.png"
    annotation_path = ROOT / "reports/annotation_metrics.png"

    if curve_path.exists() or quality_path.exists():
        tabs = st.tabs(["Кривая обучения (AL)", "Качество данных", "Разметка"])
        with tabs[0]:
            if curve_path.exists():
                st.image(str(curve_path))
            else:
                st.info("График ещё не создан")
        with tabs[1]:
            if quality_path.exists():
                st.image(str(quality_path))
            else:
                st.info("График ещё не создан")
        with tabs[2]:
            if annotation_path.exists():
                st.image(str(annotation_path))
            else:
                st.info("График ещё не создан")

    # Отчёты
    st.markdown("---")
    st.subheader("Текстовые отчёты")
    for name, path in [
        ("Quality Report", "reports/quality_report.md"),
        ("Annotation Report", "reports/annotation_report.md"),
        ("AL Report", "reports/al_report.md"),
    ]:
        p = ROOT / path
        if p.exists():
            with st.expander(name):
                st.markdown(p.read_text())

# ══════════════════════════════════════════════════════════════
# ДАТАСЕТ
# ══════════════════════════════════════════════════════════════
elif page == "📁 Датасет":
    st.header("Финальный датасет")

    labeled_path = ROOT / "data/labeled/collected_labeled.parquet"
    if not labeled_path.exists():
        st.warning("Датасет не найден. Сначала запусти пайплайн.")
        st.stop()

    df = pd.read_parquet(labeled_path)

    # Фильтры
    col1, col2 = st.columns(2)
    with col1:
        label_filter = st.selectbox("Метка", ["все"] + sorted(df["auto_label"].dropna().unique().tolist()))
    with col2:
        source_filter = st.selectbox("Источник", ["все"] + sorted(df["source"].unique().tolist()))

    filtered = df.copy()
    if label_filter != "все":
        filtered = filtered[filtered["auto_label"] == label_filter]
    if source_filter != "все":
        filtered = filtered[filtered["source"] == source_filter]

    st.info(f"Показано: **{len(filtered)}** из {len(df)} записей")
    st.dataframe(filtered[["text", "auto_label", "confidence", "source"]].head(100), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Скачать CSV", csv, "dataset.csv", "text/csv")
    with col2:
        import io
        buf = io.BytesIO()
        filtered.to_parquet(buf, index=False)
        st.download_button("⬇️ Скачать Parquet", buf.getvalue(), "dataset.parquet")
