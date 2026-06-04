import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from generator import generate_schedule, count_conflicts, ALL_SLOTS

# ---------- Настройка страницы ----------
st.set_page_config(
    page_title="Генератор расписания",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style='whitegrid')

@st.cache_resource
def load_models():
    reg = joblib.load('model_regression.pkl')
    clf = joblib.load('model_classification.pkl')
    return reg, clf


@st.cache_data
def load_data():
    courses = pd.read_csv('courses_preprocessed.csv')
    rooms = pd.read_csv('rooms.csv')
    return courses, rooms


@st.cache_data(show_spinner=False)
def cached_generate(use_ml: bool, seed: int):
    reg, clf = load_models()
    courses, rooms = load_data()
    schedule, unplaced = generate_schedule(
        courses, ALL_SLOTS, rooms, reg, clf, use_ml=use_ml, seed=seed
    )
    return schedule, unplaced


st.title("генератор расписания учебных заведений")
st.markdown(
    "интеллектуальное приложение для построения расписания на основе трёх обученных ML-моделей: регрессии, классификации и кластеризации."
)
with st.sidebar:
    st.header("параметры")

    use_ml = st.toggle(
        "использовать ML-модели",
        value=True,
        help="С ML: генератор ранжирует размещения по реалистичности от классификатора и использует предсказания регрессии для подбора размера аудитории. "
             "Без ML: жадный бейзлайн со случайным порядком слотов."
    )

    seed = st.number_input(
        "Random seed",
        min_value=0, max_value=9999, value=42, step=1,
        help="для воспроизводимости, влияет только на бейзлайн без ML."
    )

    if st.button("сгенерировать", type="primary", use_container_width=True):
        with st.spinner("генерация расписания…" + (" может занять до минуты при первом запуске с ML." if use_ml else "")):
            schedule, unplaced = cached_generate(use_ml, seed)
        st.session_state['schedule'] = schedule
        st.session_state['unplaced'] = unplaced
        st.session_state['use_ml'] = use_ml

    st.divider()
    st.subheader("загруженные модели")
    st.markdown(
        "- **регрессия:** Ridge - предсказание enrollment  \n"
        "- **классификация:** Gradient Boosting - реалистичность  \n"
        "- **кластеризация:** KMeans (k=2) - группировка курсов"
    )

# Если расписание ещё не сгенерировано — генерируем по умолчанию (с ML)
if 'schedule' not in st.session_state:
    with st.spinner("первоначальная генерация расписания… (может занять до минуты)"):
        schedule, unplaced = cached_generate(True, 42)
    st.session_state['schedule'] = schedule
    st.session_state['unplaced'] = unplaced
    st.session_state['use_ml'] = True

schedule = st.session_state['schedule']
unplaced = st.session_state['unplaced']
used_ml = st.session_state['use_ml']

courses_full, rooms_df = load_data()
total = len(courses_full)

# ---------- KPI-карточки ----------
conflicts = count_conflicts(schedule)
avg_realism = schedule['realism_score'].mean()
avg_util = (schedule['predicted_enrollment'] / schedule['room_capacity']).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("размещено курсов", f"{len(schedule)} / {total}",
          delta=f"{len(schedule) - total}" if len(schedule) < total else None,
          delta_color="inverse" if len(schedule) < total else "off")
c2.metric("конфликтов", conflicts, delta_color="inverse")
c3.metric("средняя реалистичность", f"{avg_realism:.3f}")
c4.metric("средняя утилизация", f"{avg_util:.3f}")

mode_label = "с использованием ML-моделей" if used_ml else "бейзлайн без ML"
st.caption(f"текущий режим: **{mode_label}**")

# ---------- Вкладки ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 расписание", "🗓 Сетка недели", "🏛 По корпусам",
    "сравнение с/без ML", "ℹ️ О моделях"
])

# === Вкладка 1: таблица расписания с фильтрами ===
with tab1:
    st.subheader("сгенерированное расписание")

    col1, col2, col3 = st.columns(3)
    subj_options = ['все'] + sorted(schedule['subject'].unique().tolist())
    day_options = ['все'] + sorted(schedule['day_pattern'].unique().tolist())
    build_options = ['все'] + sorted(schedule['building_name'].unique().tolist())

    sel_subj = col1.selectbox("предмет", subj_options)
    sel_day = col2.selectbox("дни недели", day_options)
    sel_build = col3.selectbox("корпус", build_options)

    filt = schedule.copy()
    if sel_subj != 'все':
        filt = filt[filt['subject'] == sel_subj]
    if sel_day != 'все':
        filt = filt[filt['day_pattern'] == sel_day]
    if sel_build != 'все':
        filt = filt[filt['building_name'] == sel_build]

    show_cols = ['course_code', 'subject', 'instructor_id', 'day_pattern',
                 'hour', 'duration_min', 'room_id', 'building_name',
                 'room_capacity', 'predicted_enrollment', 'realism_score']
    show = filt[show_cols].copy()
    show['predicted_enrollment'] = show['predicted_enrollment'].round(1)
    show['realism_score'] = show['realism_score'].round(3)
    show = show.sort_values(['day_pattern', 'hour']).reset_index(drop=True)

    st.caption(f"показано: {len(show)} из {len(schedule)} записей")
    st.dataframe(show, use_container_width=True, height=500)

    csv = show.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "⬇️ скачать выбранные записи в CSV",
        csv, "schedule.csv", "text/csv"
    )

    if unplaced:
        st.warning(f"не удалось разместить {len(unplaced)} курсов: {', '.join(unplaced[:10])}"
                   + ("…" if len(unplaced) > 10 else ""))

# === Вкладка 2: heatmap дни × часы ===
with tab2:
    st.subheader("загрузка недели по дням и часам")
    st.caption("сколько пар начинается в каждый час каждого дня. "
               "если у курса паттерн `Mon,Wed,Fri`, он засчитывается во все три дня.")

    rows = []
    for _, r in schedule.iterrows():
        for d in r['day_pattern'].split(','):
            rows.append({'day': d, 'hour': r['hour']})
    long_df = pd.DataFrame(rows)

    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    pivot = long_df.groupby(['hour', 'day']).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'пар, начинающихся в этот час'})
    ax.set_title('распределение начал пар по дням и часам')
    ax.set_xlabel('день недели')
    ax.set_ylabel('час начала')
    st.pyplot(fig)

    st.markdown("**топ загруженных слотов:**")
    top_slots = (schedule.groupby(['day_pattern', 'hour']).size()
                 .sort_values(ascending=False).head(10).reset_index())
    top_slots.columns = ['дни', 'час', 'пар']
    st.dataframe(top_slots, use_container_width=True, hide_index=True)

# === Вкладка 3: распределение предметов по корпусам ===
with tab3:
    st.subheader("соответствие предметов и корпусов")
    st.caption("на этой вкладке видно, насколько генератор корректно поселил предметы "
               "в «правильные» корпуса. с включённым ML математика должна оказаться в "
               "Mathematics Center / Anderson Hall, биология — в Biology Building и т.п.")

    top_subjects = schedule['subject'].value_counts().head(12).index
    pivot_sb = schedule[schedule['subject'].isin(top_subjects)] \
        .groupby(['subject', 'building_name']).size().unstack(fill_value=0)
    pivot_sb = pivot_sb.loc[:, pivot_sb.sum() > 0]
    pivot_sb = pivot_sb.reindex(top_subjects)

    fig2, ax2 = plt.subplots(figsize=(16, 7))
    sns.heatmap(pivot_sb, annot=True, fmt='d', cmap='YlGnBu', ax=ax2,
                cbar_kws={'label': 'курсов в этом корпусе'})
    ax2.set_title('сколько курсов каждого предмета попало в каждый корпус (топ-12 предметов)')
    ax2.set_xlabel('корпус')
    ax2.set_ylabel('предмет')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)

# === Вкладка 4: сравнение режимов ===
with tab4:
    st.subheader("сравнение работы генератора с ML и без ML")
    st.caption("запускаем генератор в обоих режимах с одним и тем же seed и сравниваем итог.")

    with st.spinner("Считаем оба варианта…"):
        sched_ml, _ = cached_generate(True, seed)
        sched_base, _ = cached_generate(False, seed)

    cmp_rows = []
    for label, s in [("С ML", sched_ml), ("Без ML", sched_base)]:
        cmp_rows.append({
            'режим': label,
            'размещено': len(s),
            'конфликтов': count_conflicts(s),
            'средняя реалистичность': round(s['realism_score'].mean(), 4),
            'средняя утилизация': round(
                (s['predicted_enrollment'] / s['room_capacity']).mean(), 3)
        })
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(['С ML', 'Без ML'], [len(sched_ml), len(sched_base)],
                color=['#4C72B0', '#C44E52'])
    axes[0].set_title('размещено курсов')
    axes[0].set_ylim(0, total + 10)

    axes[1].bar(['С ML', 'Без ML'],
                [sched_ml['realism_score'].mean(), sched_base['realism_score'].mean()],
                color=['#4C72B0', '#C44E52'])
    axes[1].set_title('средняя реалистичность')
    axes[1].set_ylim(0, 1)

    util_ml = (sched_ml['predicted_enrollment'] / sched_ml['room_capacity']).mean()
    util_base = (sched_base['predicted_enrollment'] / sched_base['room_capacity']).mean()
    axes[2].bar(['С ML', 'Без ML'], [util_ml, util_base], color=['#4C72B0', '#C44E52'])
    axes[2].set_title('средняя утилизация аудиторий')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    st.pyplot(fig3)

    st.info(
        "**интерпретация:** ML заметно повышает реалистичность размещения "
        "(математика в правильных корпусах, разумное время), ценой небольшого "
        "снижения плотности упаковки аудиторий. Без ML генератор оптимизирует "
        "только заполняемость, не заботясь о правдоподобии."
    )

# === Вкладка 5: информация о моделях ===
with tab5:
    st.subheader("ML-модели, лежащие в основе генератора")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Регрессия")
        st.markdown("**Алгоритм:** Ridge Regression")
        st.markdown("**Задача:** предсказать `current_enrollment` — сколько студентов "
                    "запишется на курс — по его признакам.")
        st.markdown("**Применение в генераторе:** подбор размера аудитории.")
        st.metric("R² (RepeatedKFold)", "≈ 0.66")
        st.metric("RMSE", "≈ 16.3")

    with col2:
        st.markdown("### 🔍 Классификация")
        st.markdown("**Алгоритм:** Gradient Boosting Classifier")
        st.markdown("**Задача:** отличить реалистичное сочетание признаков курса "
                    "и его размещения (предмет, время, длительность, корпус) от "
                    "случайно перемешанного.")
        st.markdown("**Применение в генераторе:** ранжирование допустимых размещений "
                    "по «правдоподобию».")
        st.metric("ROC-AUC", "≈ 0.92")
        st.metric("F1", "≈ 0.85")

    with col3:
        st.markdown("### 🧩 Кластеризация")
        st.markdown("**Алгоритм:** KMeans (k=2)")
        st.markdown("**Задача:** сгруппировать курсы по характеру: «длинные занятия "
                    "малыми группами» vs «короткие массовые курсы».")
        st.markdown("**Применение:** метка кластера — дополнительный признак "
                    "в регрессии и классификации.")
        st.metric("Силуэт", "≈ 0.52")
        st.metric("Кластеров", "2")

    st.divider()
    st.markdown(
        "**Архитектура генератора.** Жёсткие ограничения (один преподаватель — "
        "одна пара одновременно, одна аудитория — один курс, размер ≥ числа студентов) "
        "соблюдаются явной проверкой. ML-модели не заменяют эту проверку, а **ранжируют** "
        "допустимые варианты по неявным закономерностям, выученным из реальных расписаний."
    )