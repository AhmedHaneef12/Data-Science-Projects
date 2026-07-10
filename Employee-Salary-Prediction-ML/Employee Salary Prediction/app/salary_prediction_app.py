"""
Task #06 - Employee Salary Prediction
Streamlit App: AI-Powered Salary Prediction Tool
Dataset : Employee Salary Dataset — Kaggle
Source  : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data

Run: streamlit run app/salary_prediction_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json, re
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E6DA4 100%);
        padding: 22px 32px; border-radius: 12px; color: white;
        margin-bottom: 20px; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p  { margin: 6px 0 0; opacity: 0.85; font-size: 0.95rem; }

    .salary-box {
        background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
        border-radius: 12px; padding: 26px; text-align: center;
        color: white; margin: 14px 0;
    }
    .salary-box .label  { font-size: 0.95rem; opacity: 0.9; letter-spacing: 0.05em; }
    .salary-box .amount { font-size: 2.6rem; font-weight: 800; margin: 5px 0; }
    .salary-box .sub    { font-size: 0.9rem; opacity: 0.85; }

    .insight-card {
        background: #F0F4F8; border-left: 4px solid #E07B39;
        border-radius: 6px; padding: 13px 16px; margin: 8px 0; font-size: 0.93rem;
    }
    .warn-box {
        background: #FFF9E6; border-left: 4px solid #F0A500;
        border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem;
        color: #7D5A00;
    }
    .section-header {
        color: #1E3A5F; font-weight: 700; font-size: 1.05rem;
        border-bottom: 2px solid #E07B39; padding-bottom: 4px; margin: 14px 0 8px;
    }
    .stButton > button {
        background: #E07B39 !important; color: white !important;
        border: none !important; border-radius: 8px !important;
        padding: 10px 28px !important; font-weight: 700 !important;
        font-size: 1rem !important; width: 100%;
    }
    .stButton > button:hover { background: #C0612A !important; }
    .data-source {
        background: #EBF5FF; border: 1px solid #BDD7EE;
        border-radius: 6px; padding: 8px 14px; font-size: 0.82rem; color: #1E3A5F;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base  = Path(__file__).parent.parent
    model = joblib.load(base / 'models' / 'best_model.pkl')
    meta  = joblib.load(base / 'models' / 'model_metadata.pkl')
    return model, meta

try:
    model, meta = load_model()
except Exception as e:
    st.error(f"Model not found — run `03_model_training.py` first.\n\n{e}")
    st.stop()

# ── Constants (from training data) ────────────────────────────────────────────
DEPARTMENTS     = ['Finance', 'HR', 'IT', 'Marketing', 'Operations']
EDUCATION_LEVELS= ['High School', 'Bachelor', 'Master', 'PhD']
CITIES          = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai']
GENDERS         = ['Female', 'Male']

DEPT_AVG = {'Finance': 67262, 'HR': 77202, 'IT': 79345, 'Marketing': 96431, 'Operations': 88384}
CITY_AVG = {'Bangalore': 99004, 'Chennai': 57380, 'Delhi': 80620, 'Hyderabad': 77175, 'Mumbai': 82875}
EDU_AVG  = {'High School': 83676, 'Bachelor': 82442, 'Master': 86258, 'PhD': 72944}

# ── AI Insights ────────────────────────────────────────────────────────────────
def get_ai_insights(emp: dict, salary: float, api_key: str = "") -> tuple:
    """Claude API insight, falls back to rule-based."""
    if api_key and api_key.strip().startswith("sk-ant-"):
        try:
            import urllib.request
            prompt = f"""You are a HR compensation expert. An employee profile:
- Age: {emp['Age']}, Experience: {emp['Experience_Years']} years
- Education: {emp['Education_Level']}, Department: {emp['Department']}
- Gender: {emp['Gender']}, City: {emp['City']}
- ML Predicted Monthly Salary: ₹{salary:,.0f}

Respond ONLY in this exact JSON (no markdown fences):
{{
  "salary_range": "e.g. ₹X – ₹Y per month",
  "top_factors": ["factor1","factor2","factor3"],
  "career_tips": ["tip1","tip2","tip3"],
  "skill_recommendations": ["skill1","skill2","skill3"],
  "market_insight": "one sentence market context"
}}"""
            payload = json.dumps({
                "model": "claude-sonnet-4-6", "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key.strip(),
                         "anthropic-version": "2023-06-01"}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                text = re.sub(r'^```json\s*|\s*```$', '', data['content'][0]['text'].strip())
                return json.loads(text), "api"
        except Exception:
            pass
    return rule_insights(emp, salary), "rule"


def rule_insights(emp: dict, salary: float) -> dict:
    dept     = emp['Department']
    edu      = emp['Education_Level']
    exp      = emp['Experience_Years']
    city     = emp['City']
    dept_avg = DEPT_AVG.get(dept, 80000)
    city_avg = CITY_AVG.get(city, 80000)

    # Salary range ±10%
    rng = f"₹{salary*0.90:,.0f} – ₹{salary*1.10:,.0f} per month"

    # Top factors
    factors = [
        f"Years of experience ({exp} yrs) — strongest salary predictor (47% RF importance)",
        f"Department ({dept}) — avg ₹{dept_avg:,}/mo in this dataset",
        f"City ({city}) — avg ₹{city_avg:,}/mo; location affects cost-of-living adjustments",
    ]

    # Career tips
    tips = []
    if exp < 5:
        tips.append("In your early career, focus on building a strong portfolio and gaining certifications.")
    elif exp < 10:
        tips.append(f"At {exp} years, target team-lead or specialist roles to accelerate into ₹1L+ salary range.")
    else:
        tips.append(f"With {exp} years experience, pursue senior or managerial roles to maximise compensation.")
    if salary < dept_avg * 0.85:
        tips.append(f"Your salary is below the {dept} dept average (₹{dept_avg:,}). A performance discussion or internal move may help.")
    else:
        tips.append(f"Your salary is competitive within the {dept} department.")
    if edu in ['High School', 'Bachelor']:
        next_edu = 'Master' if edu == 'Bachelor' else 'Bachelor'
        tips.append(f"Upgrading from {edu} to {next_edu} may increase salary by ₹3,000–₹15,000/mo.")
    else:
        tips.append(f"{edu} qualification is strong; consider professional certifications for a further boost.")

    # Skills by department
    skill_map = {
        'IT':         ['Cloud (AWS/Azure)', 'Python & System Design', 'DevOps & CI/CD'],
        'Marketing':  ['Digital Marketing & SEO', 'Data Analytics', 'Content Strategy'],
        'Finance':    ['Financial Modelling', 'Power BI / Tableau', 'CFA Certification'],
        'Operations': ['Supply Chain Optimisation', 'ERP Systems (SAP)', 'Project Management (PMP)'],
        'HR':         ['People Analytics', 'HR Tech Platforms', 'Talent Acquisition Strategy'],
    }
    skills = skill_map.get(dept, ['Communication', 'Data Literacy', 'Project Management'])

    market = (f"The {dept} sector in {city} shows growing demand for skilled professionals; "
              f"salary benchmarks range from ₹{city_avg*0.7:,.0f} to ₹{city_avg*1.4:,.0f}/mo depending on experience.")

    return {"salary_range": rng, "top_factors": factors,
            "career_tips": tips, "skill_recommendations": skills, "market_insight": market}


# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>💼 Employee Salary Predictor</h1>
    <p>AI-Powered Monthly Salary Prediction with Career Insights | Gexton Education — Task #06</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="data-source">
    📊 <strong>Dataset:</strong> Employee Salary Dataset — Kaggle &nbsp;|&nbsp;
    <a href="https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data" target="_blank">
    kaggle.com/datasets/prince7489/employee-salary-dataset</a>
    &nbsp;|&nbsp; 50 records across 5 departments & 5 cities (India)
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Info")
    st.info(f"**Best Model:** {meta['model_name']}\n\n**R² (Test):** {meta['r2']:.4f}\n\n**RMSE:** ₹{meta['rmse']:,.0f}/mo")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Note:</strong> R² is low due to the small dataset (50 records).
    Predictions indicate trends only. Accuracy will improve significantly with more data.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🤖 AI Insights (Optional)")
    st.caption("Enter your Anthropic API key for Claude-powered insights. Leave blank for rule-based insights.")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.markdown("---")
    st.markdown("### 📊 Dataset Summary")
    st.markdown("- **Source:** Kaggle\n- **Records:** 50 employees\n- **Departments:** 5\n- **Cities:** 5 Indian cities\n- **Target:** Monthly Salary (₹)")

# ── Main ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="section-header">👤 Employee Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age  = st.slider("Age", 20, 60, 30)
        edu  = st.selectbox("Education Level", EDUCATION_LEVELS, index=1)
    with c2:
        gender = st.selectbox("Gender", GENDERS)
        exp    = st.slider("Years of Experience", 0, 20, 5)

    dept = st.selectbox("Department", DEPARTMENTS)
    city = st.selectbox("City", CITIES)

    st.markdown("")
    predict_btn = st.button("🔍 Predict Monthly Salary", use_container_width=True)

with col_right:
    if predict_btn:
        inp = {'Age': age, 'Experience_Years': exp, 'Education_Level': edu,
               'Department': dept, 'Gender': gender, 'City': city}
        pred = float(model.predict(pd.DataFrame([inp]))[0])
        pred = max(pred, 20000)

        st.markdown(f"""
        <div class="salary-box">
            <div class="label">PREDICTED MONTHLY SALARY</div>
            <div class="amount">₹{pred:,.0f}</div>
            <div class="sub">Annual ≈ ₹{pred*12:,.0f} &nbsp;|&nbsp; Weekly ≈ ₹{pred/4.3:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        dept_avg = DEPT_AVG.get(dept, 80000)
        diff_pct = (pred - dept_avg) / dept_avg * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("vs Dept Average", f"₹{dept_avg:,}", f"{diff_pct:+.1f}%")
        m2.metric("Experience Band",
                  "Senior" if exp >= 12 else "Mid-level" if exp >= 5 else "Junior", f"{exp} yrs")
        m3.metric("City Avg", f"₹{CITY_AVG.get(city,80000):,}", city)

        # AI Insights
        st.markdown('<div class="section-header">🤖 AI Career Insights</div>', unsafe_allow_html=True)
        with st.spinner("Generating insights..."):
            insights, src = get_ai_insights(inp, pred, api_key)

        if src == "api":
            st.success("✨ Powered by Claude AI (Anthropic)")
        else:
            st.info("📊 Powered by EDA-based intelligent rule engine")

        st.markdown(f'<div class="insight-card"><strong>📈 Expected Salary Range</strong><br>{insights["salary_range"]}</div>',
                    unsafe_allow_html=True)
        with st.expander("🔑 Key Salary Factors", expanded=True):
            for f in insights['top_factors']: st.markdown(f"• {f}")
        with st.expander("🚀 Career Improvement Tips", expanded=True):
            for t in insights['career_tips']: st.markdown(f"• {t}")
        with st.expander("🎯 Skills for Higher Salary", expanded=True):
            for s in insights['skill_recommendations']: st.markdown(f"• {s}")
        st.markdown(f'<div class="insight-card"><strong>🌐 Market Insight</strong><br>{insights["market_insight"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#7F8C8D;">
            <div style="font-size:4rem;">💼</div>
            <h3 style="color:#1E3A5F;">Ready to Predict</h3>
            <p>Fill in the employee details on the left and click<br><strong>Predict Monthly Salary</strong></p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95A5A6;font-size:0.82rem;'>"
    "Gexton Education — Data Science Internship Program &nbsp;|&nbsp; Task #06 &nbsp;|&nbsp; "
    "Supervised by Sir Muhammad Arham MH &nbsp;|&nbsp; "
    "Dataset: <a href='https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data'>"
    "Kaggle</a></div>",
    unsafe_allow_html=True
)
