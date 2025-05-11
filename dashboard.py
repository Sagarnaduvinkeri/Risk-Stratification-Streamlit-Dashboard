import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import shape
from io import BytesIO
import os

# Set page configuration
st.set_page_config(
    page_title="Healthcare Risk Stratification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0e4c92;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e88e5;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>Healthcare Risk Stratification Dashboard</div>", unsafe_allow_html=True)

# --- Load CSV from Google Drive ---
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?export=download&id=1yb0PcQtMn-cOGWtrZmabUbRrtjjAGm_S"
        df = pd.read_csv(url, encoding="ISO-8859-1")
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Drive: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.markdown("<div class='sub-header'>Filters</div>", unsafe_allow_html=True)

    # Sidebar filters
    risk_levels = ["All"] + sorted(df["CURRENT_RISK_LEVEL"].dropna().unique().tolist())
    selected_risk_level = st.sidebar.selectbox("Risk Level", risk_levels)

    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    selected_gender = st.sidebar.selectbox("Gender", genders)

    #counties = ["All"] + sorted([str(x) for x in df["HOME_COUNTY"].dropna().unique()])
    #selected_county = st.sidebar.selectbox("County", counties)

    # Filter by state
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
    #if selected_county != "All":
     #   filtered_df = filtered_df[filtered_df["HOME_COUNTY"] == selected_county]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["HOME_STATE"] == selected_state]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_range[0]) &
        (filtered_df["Age"] <= age_range[1]) &
        (filtered_df["Count of Chron Disease"] >= chronic_range[0]) &
        (filtered_df["Count of Chron Disease"] <= chronic_range[1])
    ]

    st.sidebar.markdown(f"**Total Members:** {len(filtered_df)}")

    # Download filtered data
    def generate_excel():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Filtered_Data")
        return output.getvalue()

    st.sidebar.download_button(
        "Download Filtered Data",
        data=generate_excel(),
        file_name="filtered_risk_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🏥 Clinical Metrics", "💲 Financial Metrics", "🔍 Member Details"
    ])
    # --- Tab 1: Overview ---
    with tab1:
        st.markdown("<div class='sub-header'>Population Overview</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
        with col2:
            st.metric("Avg Chronic Conditions", f"{filtered_df['Count of Chron Disease'].mean():.1f}")
        with col3:
            st.metric("Avg Prospective Risk", f"{filtered_df['PROSP_TOTAL_RISK'].mean():.2f}")
        with col4:
            rising_risk_pct = (filtered_df['RISING_RISK_FLAG'].sum() / len(filtered_df)) * 100
            st.metric("Rising Risk %", f"{rising_risk_pct:.1f}%")

        # Risk Level Distribution
        st.markdown("<div class='section-header'>Risk Level Member Distribution</div>", unsafe_allow_html=True)

        # Step 1: Clean risk level column separately without overwriting original
        clean_risk = pd.to_numeric(filtered_df['CURRENT_RISK_LEVEL'], errors='coerce').astype('Int64')

        # Step 2: Define valid risk levels
        valid_levels = list(range(1, 9)) + [99]

        # Step 3: Filter rows with valid risk levels only
        filtered_risk_df = filtered_df[clean_risk.isin(valid_levels)].copy()
        filtered_risk_df['Risk_Label'] = clean_risk[clean_risk.isin(valid_levels)].astype(str)

        # Optional: Label 99 as '99 (Unclassified)' for clarity
        filtered_risk_df['Risk_Label'] = filtered_risk_df['Risk_Label'].replace({'99': '99 (Unclassified)'})

        # Step 4: Count members per risk level (in desired order)
        risk_order = [str(i) for i in range(1, 9)] + ['99 (Unclassified)']
        risk_count = (
            filtered_risk_df['Risk_Label']
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )
        risk_count.columns = ['Risk Level', 'Count']

        # Step 5: Plot the bar chart
        fig = px.bar(
            risk_count,
            x='Risk Level',
            y='Count',
            text='Count',
            color_discrete_sequence=['#1e88e5']
        )
        fig.update_layout(
            height=400,
            xaxis_type='category',
            xaxis_title='Risk Level',
            yaxis_title='Member Count'
        )
        st.plotly_chart(fig, use_container_width=True)


# Gender and Age distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-header'>Gender Distribution</div>", unsafe_allow_html=True)
            gender_count = filtered_df['Gender'].value_counts().reset_index()
            gender_count.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_count, 
                names='Gender', 
                values='Count',
                color='Gender',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='section-header'>Age Distribution</div>", unsafe_allow_html=True)
            fig = px.histogram(
                filtered_df, 
                x='Age',
                nbins=10,
                color_discrete_sequence=['#1e88e5']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk vs Chronic Disease
        st.markdown("<div class='section-header'>Prospective Risk vs Chronic Disease Count</div>", unsafe_allow_html=True)
        filtered_df = filtered_df.dropna(subset=['PROSP_IP_RISK'])

        fig = px.scatter(
            filtered_df,
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
        top_dx = filtered_df['DX_TOP_DESC'].value_counts().reset_index()
        top_dx.columns = ['Diagnosis', 'Count']
        top_dx = top_dx.head(10)
        
        fig = px.bar(
            top_dx,
            x='Count',
            y='Diagnosis',
            orientation='h',
            color_discrete_sequence=['#1e88e5'] 
        )
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Chronic condition prevalence
        st.markdown("<div class='section-header'>Chronic Condition Prevalence</div>", unsafe_allow_html=True)
        
        # Select relevant condition columns
        condition_cols = [col for col in df.columns if col.startswith('_') and col.endswith('_IND')]
        condition_names = [col.replace('_', '').replace('_IND', '') for col in condition_cols]
        
        # Calculate prevalence
        prevalence_data = []
        for col, name in zip(condition_cols, condition_names):
            if col in filtered_df.columns:
                count = filtered_df[col].sum()
                prevalence_data.append({
                    'Condition': name,
                    'Count': count,
                    'Prevalence': count / len(filtered_df) * 100
                })
        
        prevalence_df = pd.DataFrame(prevalence_data)
        prevalence_df = prevalence_df.sort_values('Prevalence', ascending=False)
        
        fig = px.bar(
            prevalence_df,
            x='Condition',
            y='Prevalence',
            color='Prevalence',
            text='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cancer types if applicable
        cancer_df = filtered_df[filtered_df['CANCER_ACTIVE_IND'] == 1].copy()
        if not cancer_df.empty:
            st.markdown("<div class='section-header'>Cancer Types</div>", unsafe_allow_html=True)
            
            cancer_types = cancer_df['CANCER_TYPE'].value_counts().reset_index()
            cancer_types.columns = ['Cancer Type', 'Count']
            
            fig = px.pie(
                cancer_types,
                names='Cancer Type',
                values='Count',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Laboratory metrics
        st.markdown("<div class='section-header'>Key Laboratory Metrics</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # A1C distribution for members with values
            a1c_df = filtered_df[filtered_df['A1C_LATEST_RESULT'].notna()].copy()
            
            # Convert string values to numeric where possible
            a1c_df['A1C_NUMERIC'] = pd.to_numeric(a1c_df['A1C_LATEST_RESULT'], errors='coerce')
            
            if not a1c_df.empty and a1c_df['A1C_NUMERIC'].notna().sum() > 0:
                fig = px.histogram(
                    a1c_df[a1c_df['A1C_NUMERIC'].notna()],
                    x='A1C_NUMERIC',
                    nbins=10,
                    color_discrete_sequence=['#ff7043'],
                    title="A1C Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough A1C data available for visualization")
        
        with col2:
            # BMI distribution
            bmi_df = filtered_df[filtered_df['BMI'].notna()].copy()
            
            if not bmi_df.empty:
                fig = px.histogram(
                    bmi_df,
                    x='BMI',
                    nbins=10,
                    color_discrete_sequence=['#26a69a'],
                    title="BMI Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough BMI data available for visualization")

    # --- Tab 3: Financial Metrics ---
    with tab3:
        st.markdown("<div class='sub-header'>Financial Analysis</div>", unsafe_allow_html=True)
        
        # Risk score impact on costs
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
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        st.markdown("<div class='section-header'>Cost Distribution by Service Type</div>", unsafe_allow_html=True)
        
        # Prepare cost data
        cost_cols = ['ALWD_ER', 'ALWD_IP', 'ALWD_OFFICE', 'ALWD_OP', 'ALWD_RX', 'ALWD_OTHER']
        cost_data = filtered_df[cost_cols].mean().reset_index()
        cost_data.columns = ['Service Type', 'Average Cost']
        cost_data['Service Type'] = cost_data['Service Type'].str.replace('ALWD_', '')
        
        fig = px.pie(
            cost_data,
            names='Service Type',
            values='Average Cost',
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly trend
        st.markdown("<div class='section-header'>Quarterly Spend Trend</div>", unsafe_allow_html=True)
        
        quarters = ['ALWD_Q1', 'ALWD_Q2', 'ALWD_Q3', 'ALWD_Q4']
        quarterly_data = filtered_df[quarters].mean().reset_index()
        quarterly_data.columns = ['Quarter', 'Average Cost']
        quarterly_data['Quarter'] = quarterly_data['Quarter'].str.replace('ALWD_', 'Q')
        
        fig = px.line(
            quarterly_data,
            x='Quarter',
            y='Average Cost',
            markers=True,
            line_shape='linear',
            color_discrete_sequence=['#e91e63']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Expected PMPM change
        
        # Step 1: Clean and convert PMPM midpoint values to float
        filtered_df['PMPM_CHANGE_NUMERIC'] = filtered_df['EXPECTED_PMPM_CHANGE_MIDPOINT'] \
            .str.replace('$', '', regex=False) \
            .str.replace('(', '-', regex=False) \
            .str.replace(')', '', regex=False) \
            .str.strip() \
            .astype(float)

        # Step 2: Group by risk level
        bullet_data = filtered_df[filtered_df['CURRENT_RISK_LEVEL'].isin(list(range(1, 9)) + [99])]
        bullet_grouped = bullet_data.groupby('CURRENT_RISK_LEVEL')['PMPM_CHANGE_NUMERIC'].mean().reset_index()
        bullet_grouped['CURRENT_RISK_LEVEL'] = bullet_grouped['CURRENT_RISK_LEVEL'].astype(str)

        # Step 3: Create a bullet chart using horizontal bar plots
        fig = go.Figure()

        for _, row in bullet_grouped.iterrows():
            level = row['CURRENT_RISK_LEVEL']
            value = row['PMPM_CHANGE_NUMERIC']

            fig.add_trace(go.Bar(
                x=[value],
                y=[level],
                orientation='h',
                marker=dict(color='steelblue'),
                name=f'Risk {level}',
                hovertemplate=f'Risk Level {level}<br>PMPM Change: ${value:,.2f}<extra></extra>',
                text=f"${value:,.0f}",
                textposition='auto'
            ))

        fig.update_layout(
            title="Expected PMPM Change by Risk Level",
            xaxis_title="Expected Change in PMPM ($)",
            yaxis_title="Risk Level",
            barmode='stack',
            height=500,
            template='plotly_white',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    # --- Tab 4: Member Details ---
    with tab4:
        st.markdown("<div class='sub-header'>Member-Level Analysis</div>", unsafe_allow_html=True)
        
        # Member search
        st.markdown("<div class='section-header'>Member Search</div>", unsafe_allow_html=True)
        
        member_list = filtered_df['Member '].tolist()
        selected_member = st.selectbox("Select Member", member_list)
        
        if selected_member:
            member_data = filtered_df[filtered_df['Member '] == selected_member].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                    ">
                        <b>Basic Information</b><br>
                        Age: {member_data['Age']}<br>
                        Gender: {member_data['Gender']}<br>
                        County: {member_data['HOME_COUNTY']}<br>
                        Member: {"Yes" if member_data['NEW_MEMBER'] == 1 else "No"}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                    ">
                        <b>Risk Profile</b><br>
                        Current Risk Level: {member_data['CURRENT_RISK_LEVEL']}<br>
                        Prospective Risk: {member_data['PROSP_TOTAL_RISK']:.2f}<br>
                        ER Risk: {member_data['PROSP_ER_RISK']:.4f}<br>
                        IP Risk: {member_data['PROSP_IP_RISK']:.4f}<br>
                        Rising Risk: {"Yes" if member_data['RISING_RISK_FLAG'] == 1 else "No"}
                    </div>  
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                    ">
                        <b>Utilization Summary</b><br>
                        IP Admits: {member_data['IP_ADMITS_ALL']}<br>
                        ER Visits: {member_data['ER_VISITS_ALL']}<br>
                        Rx Drug Count: {member_data['RX_DRUG_COUNT']}<br>
                        Provider Count: {member_data['NPI_COUNT']}<br>
                        Chronic Disease Count: {member_data['Count of Chron Disease']}
                    </div>
                """, unsafe_allow_html=True)
            
            # Member clinical details
            st.markdown("<div class='section-header'>Clinical Profile</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                    ">
                        <b>Primary Diagnosis</b><br>
                        Diagnosis Code: {member_data['DX_TOP_CODE']}<br>
                        Diagnosis Description: {member_data['DX_TOP_DESC']}<br>
                        Behavioral Health SPMI: {'Yes' if member_data['BH_SPMI'] == 1 else 'No'}<br>
                        Cancer Active: {'Yes' if member_data['CANCER_ACTIVE_IND'] == 1 else 'No'}<br>
                        {'Cancer Type: ' + member_data['CANCER_TYPE'] if member_data['CANCER_ACTIVE_IND'] == 1 else ''}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                    ">
                        <b>Lab Results</b><br>
                        BMI: {member_data['BMI'] if pd.notna(member_data['BMI']) else 'Not available'}<br>
                        A1C: {member_data['A1C_LATEST_RESULT'] if pd.notna(member_data['A1C_LATEST_RESULT']) else 'Not available'}<br>
                        GFR: {member_data['LAB_GFR_RESULT'] if pd.notna(member_data['LAB_GFR_RESULT']) else 'Not available'}<br>
                        Creatinine: {member_data['LAB_CREATININE_RESULT'] if pd.notna(member_data['LAB_CREATININE_RESULT']) else 'Not available'}
                    </div>
                """, unsafe_allow_html=True)
            
            # Conditions table
            st.markdown("<div class='section-header'>Chronic Conditions</div>", unsafe_allow_html=True)
            
            condition_cols = [col for col in df.columns if col.startswith('_') and col.endswith('_IND')]
            condition_data = []
            
            for col in condition_cols:
                if col in member_data and member_data[col] == 1:
                    condition_name = col.replace('_', '').replace('_IND', '')
                    condition_data.append(condition_name)
            
            if condition_data:
                # Split into three columns for better display
                cols = st.columns(3)
                condition_chunks = [condition_data[i:i + (len(condition_data) + 2) // 3] for i in range(0, len(condition_data), (len(condition_data) + 2) // 3)]
                
                for i, chunk in enumerate(condition_chunks):
                    with cols[i]:
                        for condition in chunk:
                            st.markdown(f"✓ {condition}")
            else:
                st.info("No chronic conditions recorded")
            
            # Medication information
            st.markdown("<div class='section-header'>Medication Profile</div>", unsafe_allow_html=True)
            
            med_cols = ['HEART_ACE_DRUG', 'HEART_ARB_DRUG', 'HEART_BETA_DRUG', 'HEART_MRA_DRUG', 'HEART_ENTRESTO_DRUG', 'DIAB_DRUG']
            med_data = []
            
            for col in med_cols:
                if col in member_data and pd.notna(member_data[col]) and member_data[col]:
                    med_name = member_data[col] if isinstance(member_data[col], str) else col.replace('_', ' ')
                    med_data.append(med_name)
            
            if med_data:
                # Split into three columns for better display
                cols = st.columns(3)
                med_chunks = [med_data[i:i + (len(med_data) + 2) // 3] for i in range(0, len(med_data), (len(med_data) + 2) // 3)]
                
                for i, chunk in enumerate(med_chunks):
                    with cols[i]:
                        for med in chunk:
                            st.markdown(f"💊 {med}")
            else:
                st.info("No medication information available")
            
            # Cost breakdown for the member
            st.markdown("<div class='section-header'>Cost Profile</div>", unsafe_allow_html=True)
            
            cost_cols = ['ALWD_ER', 'ALWD_IP', 'ALWD_OFFICE', 'ALWD_OP', 'ALWD_RX', 'ALWD_OTHER']
            cost_data = {col.replace('ALWD_', ''): member_data[col] for col in cost_cols if col in member_data}
            
            fig = px.bar(
                x=list(cost_data.keys()),
                y=list(cost_data.values()),
                color=list(cost_data.keys()),
                labels={'x': 'Service Type', 'y': 'Cost'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No data available. Please check if the CSV file is correctly loaded.")
