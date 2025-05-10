import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# ========== CONFIG ==========
st.set_page_config(
    page_title="Healthcare Risk Stratification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== STYLING ==========
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #0e4c92; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #1e88e5; margin-bottom: 0.5rem; }
    .section-header { font-size: 1.2rem; font-weight: bold; color: #333; margin-top: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Healthcare Risk Stratification Dashboard</div>", unsafe_allow_html=True)

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?export=download&id=1yb0PcQtMn-cOGWtrZmabUbRrtjjAGm_S"
        df = pd.read_csv(url, encoding="ISO-8859-1")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# ========== SETUP ==========
if not df.empty:
    st.sidebar.markdown("<div class='sub-header'>Filters</div>", unsafe_allow_html=True)

    risk_levels = ["All"] + sorted(df["CURRENT_RISK_LEVEL"].dropna().unique().tolist())
    selected_risk_level = st.sidebar.selectbox("Risk Level", risk_levels)

    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    selected_gender = st.sidebar.selectbox("Gender", genders)

    states = ["All"] + sorted([str(state) for state in df["HOME_STATE"].unique() if pd.notnull(state)])
    selected_state = st.sidebar.selectbox("State", states)

    chronic_range = st.sidebar.slider(
        "Chronic Disease Count",
        int(df["Count of Chron Disease"].min()),
        int(df["Count of Chron Disease"].max()),
        (int(df["Count of Chron Disease"].min()), int(df["Count of Chron Disease"].max()))
    )

    filtered_df = df.copy()
    if selected_risk_level != "All":
        filtered_df = filtered_df[filtered_df["CURRENT_RISK_LEVEL"] == selected_risk_level]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["HOME_STATE"] == selected_state]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_range[0]) &
        (filtered_df["Age"] <= age_range[1]) &
        (filtered_df["Count of Chron Disease"] >= chronic_range[0]) &
        (filtered_df["Count of Chron Disease"] <= chronic_range[1])
    ]

    st.sidebar.markdown(f"**Total Members:** {len(filtered_df)}")

    # ========== TABS ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🏥 Clinical Metrics", "💲 Financial Metrics", "🔍 Member Details", "🤖 Chatbot"
    ])

    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        st.markdown("<div class='sub-header'>Population Overview</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
        with col2: st.metric("Avg Chronic Conditions", f"{filtered_df['Count of Chron Disease'].mean():.1f}")
        with col3: st.metric("Avg Prospective Risk", f"{filtered_df['PROSP_TOTAL_RISK'].mean():.2f}")
        with col4:
            rising_risk_pct = (filtered_df['RISING_RISK_FLAG'].sum() / len(filtered_df)) * 100
            st.metric("Rising Risk %", f"{rising_risk_pct:.1f}%")

    # ========== TAB 5: CHATBOT ==========
    with tab5:
        st.markdown("<div class='sub-header'>Ask the Dashboard (AI Assistant)</div>", unsafe_allow_html=True)

        if "groq_api_key" not in st.secrets:
            st.warning("Groq API key not found. Please add it in Streamlit Cloud → Settings → Secrets.")
        else:
            user_question = st.text_input("Ask a question about the dataset:")
            if user_question:
                with st.spinner("Thinking..."):
                    try:
                        sample = filtered_df[['Age', 'Gender', 'CURRENT_RISK_LEVEL', 'Count of Chron Disease', 'PROSP_TOTAL_RISK']].head(20)
                        context_summary = sample.to_markdown(index=False)

                        headers = {
                            "Authorization": f"Bearer {st.secrets['groq_api_key']}",
                            "Content-Type": "application/json"
                        }

                        payload = {
                            "model": "mixtral-8x7b-32768",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful healthcare analyst. Use the data provided to answer user questions clearly."
                                },
                                {
                                    "role": "user",
                                    "content": f"Here is the patient data sample:\n{context_summary}\n\nNow answer this: {user_question}"
                                }
                            ],
                            "temperature": 0.5
                        }

                        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

                        if response.status_code == 200:
                            answer = response.json()["choices"][0]["message"]["content"]
                            st.success(answer)
                        else:
                            st.error("Error fetching response from Groq API.")
                    except Exception as e:
                        st.error(f"Chatbot error: {e}")
else:
    st.error("No data available. Please verify the data source link.")
