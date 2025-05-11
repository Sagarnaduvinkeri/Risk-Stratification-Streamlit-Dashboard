# 🏥 Healthcare Risk Stratification Dashboard

An interactive healthcare analytics dashboard built with **Streamlit** to support dynamic risk stratification, clinical monitoring, cost analysis, and patient-level exploration. The dashboard helps healthcare professionals proactively manage high-risk patients, allocate resources effectively, and reduce healthcare costs.


## 🎯 Objectives

- Identify high-risk healthcare members for targeted intervention.
- Visualize chronic conditions, clinical risks, and financial costs.
- Enable export of filtered member-level data.
- Integrate AI assistant for interactive analytics support.
- Empower care managers to make data-driven decisions.

---

## 📊 Dashboard Features

The app includes 4 main analytic tabs:

### 1. **📊 Population Overview**
- KPIs: Avg Age, Chronic Conditions, Risk Scores
- Risk Level Distribution
- Gender and Age Breakdown
- Chronic Disease Count vs. Prospective Risk Scatter Plot

### 2. **🏥 Clinical Metrics**
- Top Diagnoses
- Chronic Condition Prevalence
- Cancer Type Breakdown
- Lab Metrics (A1C, BMI Distributions)

### 3. **💲 Financial Metrics**
- Cost by Risk Score
- Cost Distribution by Service Type
- Quarterly Spend Trends
- Expected PMPM Cost Change by Risk Level

### 4. **🔍 Member-Level Analysis**
- Detailed member cards with:
  - Risk Profile
  - Cost Breakdown
  - Lab Results
  - Chronic Conditions
  - Medications

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Backend:** Pandas, NumPy
- **Data Source:** `risk.csv` (Google Drive hosted)

---

## 📁 File Structure

├── dashboard.py # Main Streamlit dashboard app

├── Risk_Stratification_Dashboard_Report.pdf # Project Report

├── requirements.txt # Python dependencies

└── .streamlit/


---

## 🚀 Run It Locally

```bash
git clone https://github.com/yourusername/health-risk-dashboard.git
cd health-risk-dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```
---

## 📦 requirements.txt
streamlit
pandas
numpy
plotly
matplotlib
seaborn
openpyxl
shapely
statsmodels
requests

---

## 📚 Report Summary
Key insights from the clinical and financial analysis include:

Most common chronic conditions: Hypertension, Hyperlipidemia, Obesity

Top diagnoses: Type 2 Diabetes Mellitus, Primary Hypertension

Cost drivers: Prescription drugs and inpatient services

Clinical risks: High BMI and uncontrolled A1C in diabetic patients

High-risk segments: Identified using prospective risk scores and chronic disease clustering

📄 For more details, see the full Project Report

---

## 🧠 Use Cases
Population health analytics

High-risk member identification

Financial forecasting and PMPM tracking

Patient engagement and care planning

Oncology and chronic disease prioritization

---

## 🛡️ License
This project is for academic and educational use

---

🤝 Contact
Have questions or suggestions? Connect with me:

Sagar Naduvinkeri

📧 snaduvin@buffalo.edu

