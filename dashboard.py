import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# --- Page config ---
st.set_page_config(
    page_title="Healthcare Risk Stratification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #0e4c92; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #1e88e5; margin-bottom: 0.5rem; }
    .section-header { font-size: 1.2rem; font-weight: bold; color: #333; margin-top: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Healthcare Risk Stratification Dashboard</div>", unsafe_allow_html=True)

# --- Load CSV from Google Drive ---
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

if not df.empty:
    # --- Sidebar filters ---
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

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🏥 Clinical Metrics", "💲 Financial Metrics", "🔍 Member Details", "🤖 Chatbot"
    ])
    # --- Tab 1: Overview ---
    with tab1:
        st.markdown("<div class='sub-header'>Population Overview</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
        col2.metric("Avg Chronic Conditions", f"{filtered_df['Count of Chron Disease'].mean():.1f}")
        col3.metric("Avg Prospective Risk", f"{filtered_df['PROSP_TOTAL_RISK'].mean():.2f}")
        rising_risk_pct = (filtered_df['RISING_RISK_FLAG'].sum() / len(filtered_df)) * 100
        col4.metric("Rising Risk %", f"{rising_risk_pct:.1f}%")

        # Risk level distribution
        st.markdown("<div class='section-header'>Risk Level Member Distribution</div>", unsafe_allow_html=True)
        filtered_df['Risk_Label'] = pd.to_numeric(filtered_df['CURRENT_RISK_LEVEL'], errors='coerce').astype("Int64")
        risk_count = filtered_df['Risk_Label'].value_counts().reset_index()
        risk_count.columns = ['Risk Level', 'Count']
        fig = px.bar(risk_count, x='Risk Level', y='Count', text='Count', color_discrete_sequence=['#1e88e5'])
        st.plotly_chart(fig, use_container_width=True)

        # Gender and Age distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Gender Distribution</div>", unsafe_allow_html=True)
            gender_count = filtered_df['Gender'].value_counts().reset_index()
            gender_count.columns = ['Gender', 'Count']
            fig = px.pie(gender_count, names='Gender', values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>Age Distribution</div>", unsafe_allow_html=True)
            fig = px.histogram(filtered_df, x='Age', nbins=10, color_discrete_sequence=['#1e88e5'])
            st.plotly_chart(fig, use_container_width=True)

        # Risk vs chronic conditions
        st.markdown("<div class='section-header'>Prospective Risk vs Chronic Disease Count</div>", unsafe_allow_html=True)
        fig = px.scatter(
            filtered_df.dropna(subset=['PROSP_IP_RISK']),
            x='Count of Chron Disease',
            y='PROSP_TOTAL_RISK',
            color='CURRENT_RISK_LEVEL',
            size='PROSP_IP_RISK',
            hover_data=['Member ', 'Age', 'Gender', 'DX_TOP_DESC'],
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Clinical Metrics ---
    with tab2:
        st.markdown("<div class='sub-header'>Clinical Risk Factors</div>", unsafe_allow_html=True)

        # Top diagnoses
        st.markdown("<div class='section-header'>Top Diagnoses</div>", unsafe_allow_html=True)
        top_dx = filtered_df['DX_TOP_DESC'].value_counts().head(10).reset_index()
        top_dx.columns = ['Diagnosis', 'Count']
        fig = px.bar(top_dx, x='Count', y='Diagnosis', orientation='h', color_discrete_sequence=['#1e88e5'])
        st.plotly_chart(fig, use_container_width=True)

        # Chronic condition prevalence
        st.markdown("<div class='section-header'>Chronic Condition Prevalence</div>", unsafe_allow_html=True)
        condition_cols = [col for col in df.columns if col.startswith('_') and col.endswith('_IND')]
        prevalence_data = []
        for col in condition_cols:
            if col in filtered_df.columns:
                count = filtered_df[col].sum()
                prevalence_data.append({
                    'Condition': col.replace('_IND', '').strip('_'),
                    'Count': count,
                    'Prevalence': count / len(filtered_df) * 100
                })

        prevalence_df = pd.DataFrame(prevalence_data).sort_values('Prevalence', ascending=False)
        fig = px.bar(prevalence_df, x='Condition', y='Prevalence', text='Count', color='Prevalence', color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Financial Metrics ---
    with tab3:
        st.markdown("<div class='sub-header'>Financial Analysis</div>", unsafe_allow_html=True)

        # Risk score vs cost
        st.markdown("<div class='section-header'>Risk Score Impact on Medical Costs</div>", unsafe_allow_html=True)
        fig = px.scatter(
            filtered_df,
            x='PROSP_TOTAL_RISK',
            y='ALWD_MED',
            color='CURRENT_RISK_LEVEL',
            size='IP_ADMITS_ALL',
            hover_data=['Member ', 'ALWD_IP', 'ALWD_ER', 'ALWD_RX'],
            trendline='ols',
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cost distribution
        st.markdown("<div class='section-header'>Cost Distribution by Service Type</div>", unsafe_allow_html=True)
        cost_cols = ['ALWD_ER', 'ALWD_IP', 'ALWD_OFFICE', 'ALWD_OP', 'ALWD_RX', 'ALWD_OTHER']
        cost_data = filtered_df[cost_cols].mean().reset_index()
        cost_data.columns = ['Service Type', 'Average Cost']
        cost_data['Service Type'] = cost_data['Service Type'].str.replace('ALWD_', '')
        fig = px.pie(cost_data, names='Service Type', values='Average Cost', color_discrete_sequence=px.colors.sequential.Plasma_r)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: Member Details ---
    with tab4:
        st.markdown("<div class='sub-header'>Member-Level Analysis</div>", unsafe_allow_html=True)
        member_list = filtered_df['Member '].dropna().unique().tolist()
        selected_member = st.selectbox("Select Member", member_list)

        if selected_member:
            member_data = filtered_df[filtered_df['Member '] == selected_member].iloc[0]
            st.write(member_data)

    # --- Tab 5: Chatbot (Groq-powered) ---
    with tab5:
        st.markdown("<div class='sub-header'>Ask the Dashboard (AI Assistant)</div>", unsafe_allow_html=True)

        user_question = st.text_input("Ask a question about the dataset:")

        if user_question:
            with st.spinner("Thinking..."):
                try:
                    # Use sample data (first 20 rows of key columns)
                    sample = filtered_df[['Age', 'Gender', 'CURRENT_RISK_LEVEL', 'Count of Chron Disease', 'PROSP_TOTAL_RISK']].head(20)
                    context_summary = sample.to_string(index=False)  # ✅ Avoid tabulate dependency

                    headers = {
                        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
                        "Content-Type": "application/json"
                    }

                    payload = {
                        "model": "mixtral-8x7b-32768",
                        "messages": [
                            {"role": "system", "content": "You are a helpful healthcare analyst. Answer based on the dataset provided."},
                            {"role": "user", "content": f"Here is a sample from the dataset:\n{context_summary}\n\nQuestion: {user_question}"}
                        ],
                        "temperature": 0.5
                    }

                    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

                    if response.status_code == 200:
                        answer = response.json()['choices'][0]['message']['content']
                        st.success(answer)
                    else:
                        st.error(f"Error from Groq API: {response.status_code}")
                except Exception as e:
                    st.error(f"Chatbot error: {e}")
else:
    st.error("No data available. Please check the Google Drive file or link.")
