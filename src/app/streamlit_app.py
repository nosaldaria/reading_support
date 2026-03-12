import streamlit as st
import plotly.express as px
import pandas as pd
import time
import re
from src.agents.text_analyzer import TextAnalyzerAgent
from src.agents.text_simplifier import TextSimplificationAgent

st.set_page_config(
    page_title="Reading Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .level-light { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .level-medium { background: linear-gradient(135deg, #FF9800, #F57C00); }
    .level-hard { background: linear-gradient(135deg, #f44336, #D32F2F); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agents():
    analyzer = TextAnalyzerAgent(
        model_path="models/paragraph_classifier.pkl"
    )
    simplifier = TextSimplificationAgent(
        # dictionary_path="/Users/daria/PycharmProjects/reading-support-system/src/utils/models/simple_dictionary.json",
        fillers_path="src/scripts/fillers.json",
        auto_update=True
    )
    return analyzer, simplifier

analyzer, simplifier = load_agents()

with st.sidebar:
    st.markdown("### Статус агентів")
    c1, c2 = st.columns(2)
    c1.metric("Аналізатор", "OK" if analyzer else "Error")
    c2.metric("Спрощувач", "OK" if simplifier else "Error")

    if simplifier:
        stats = simplifier.get_stats()
        st.markdown("### Словник")
        c1, c2, c3 = st.columns(3)
        c1.metric("Слів", stats.get("total_words", 0))
        c2.metric("Якісних", stats.get("high_quality_words", 0))
        c3.metric("RoBERTa", stats.get("roberta_words", 0))

st.markdown('<h1 class="main-header">Reading Support System</h1>', unsafe_allow_html=True)
st.markdown("Аналіз складності тексту та автоматичне спрощення")
st.markdown("---")

text_input = st.text_area(
    "Введіть український текст:",
    height=250,
    placeholder="Введіть текст для аналізу..."
)

if st.button("Аналізувати та спростити"):
    if not text_input.strip():
        st.warning("Введіть текст для аналізу")
    else:
        with st.spinner("Обробляємо текст..."):
            start_time = time.time()

            analysis = analyzer.analyze(text_input)
            predicted_level = analysis.get("predicted_level", "середній")

            simplified_text = simplifier.simplify(text_input, predicted_level)

            elapsed = time.time() - start_time

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Аналіз складності")

                c1, c2, c3 = st.columns(3)
                c1.metric("Рівень", predicted_level.upper())
                c2.metric("Впевненість", f"{analysis.get('confidence', 0):.0%}")
                c3.metric("Слів", analysis.get("stats", {}).get("ukr_words", 0))

                if "level_probabilities" in analysis:
                    df = pd.DataFrame(
                        analysis["level_probabilities"].items(),
                        columns=["Рівень", "Ймовірність"]
                    )
                    st.plotly_chart(
                        px.bar(df, x="Рівень", y="Ймовірність"),
                        use_container_width=True
                    )

            with col2:
                st.markdown("### Спрощений текст")

                st.markdown("**Оригінал:**")
                st.info(text_input)

                st.markdown("**Після обробки:**")
                st.success(simplified_text)

                stats = simplifier.get_stats()
                st.metric(
                    "Видалено філерних слів",
                    stats.get("removed_fillers", 0)
                )

            st.success(f"Готово за {elapsed:.2f} с")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:1rem">
    Reading Support System
</div>
""", unsafe_allow_html=True)
