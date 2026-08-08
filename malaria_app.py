"""
MalariaCheck — Malaria Symptom Risk Assessment System

A machine-learning-assisted screening tool that estimates the likelihood of
malaria from reported symptoms. It is a pre-diagnostic aid: it never replaces
laboratory testing (RDT / microscopy) or professional medical care.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- page setup

st.set_page_config(
    page_title="MalariaCheck — Symptom Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .app-header {
        background: linear-gradient(120deg, #0b5d56, #0f766e 60%, #14919b);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 2.2rem; }
    .app-header p { margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.92; }
    .sub-header {
        font-size: 1.4rem;
        color: #0f766e;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e3e8e8;
        border-top: 4px solid #0f766e;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.95rem; color: #5b6b6a; font-weight: 600; }
    .metric-card h2 { margin: 0.3rem 0 0 0; font-size: 1.9rem; color: #143d38; }
    .result-positive {
        background-color: #fdecea;
        color: #7a1d12;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #c0392b;
        margin: 1rem 0;
    }
    .result-positive h3 { color: #a93226; }
    .result-negative {
        background-color: #eaf7ee;
        color: #14532d;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #1e8449;
        margin: 1rem 0;
    }
    .result-negative h3 { color: #1e8449; }
    .danger-banner {
        background-color: #c0392b;
        color: #ffffff;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.05rem;
    }
    .danger-banner a { color: #ffe3de; }
    .disclaimer {
        background-color: #fff8e6;
        color: #c0392b;
        border: 1px solid #f0dfae;
        border-left: 6px solid #d4a017;
        padding: 1rem 1.3rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .info-box {
        background-color: #eaf3fb;
        color: #12405e;
        padding: 1rem 1.3rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 6px solid #2471a3;
    }
    .privacy-note { color: #7f8c8b; font-size: 0.88rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ constants

# Order matches the training data columns.
SYMPTOMS = {
    'Fever': ('Fever',
              'Raised body temperature (≥ 37.5 °C), often in cycles of chills, '
              'high fever and sweating'),
    'Headache': ('Headache', 'Persistent or throbbing head pain'),
    'Abdominal_Pain': ('Abdominal pain', 'Pain or discomfort in the belly'),
    'General_Body_Malaise': ('General body weakness',
                             'Feeling generally unwell, weak or exhausted'),
    'Dizziness': ('Dizziness', 'Light-headedness or feeling faint'),
    'Vomiting': ('Vomiting', 'Throwing up, or repeated nausea'),
    'Confusion': ('Confusion',
                  'Disorientation, drowsiness or difficulty staying alert — '
                  'a possible sign of severe malaria'),
    'Backache': ('Back pain', 'Aching or stiffness in the back'),
    'Chest_Pain': ('Chest pain', 'Pain or tightness in the chest'),
    'Coughing': ('Cough', 'Dry or productive cough'),
    'Joint_Pain': ('Joint / muscle pain', 'Aching joints or muscles'),
}

# Symptoms that can indicate severe malaria and need urgent in-person care.
DANGER_SIGNS = {
    'Confusion': 'confusion or unusual drowsiness',
    'Vomiting': 'repeated vomiting (risk of dehydration)',
}

DISCLAIMER_HTML = """
<div class="disclaimer">
    <strong>⚕️ Medical disclaimer:</strong> This tool provides a preliminary,
    symptom-based risk estimate only. It is <strong>not</strong> a medical
    diagnosis. Malaria can only be confirmed with a laboratory test (rapid
    diagnostic test or microscopy). If you feel unwell — whatever this tool
    says — visit your nearest health facility. Untreated malaria can become
    life-threatening within 24 hours.
</div>
"""

PAGES = ["Home", "Self-Assessment", "Data Insights", "Model Performance",
         "Decision Tree", "Help & About"]

# ------------------------------------------------------------- data and model


@st.cache_data
def load_data():
    df = pd.read_csv('Malaria_Dataset.csv')

    # Keep only the columns the model can use: age, symptoms and the label.
    drop_cols = ['IP_Number', 'DOA', 'Discharge_Date', 'Primary_Code',
                 'Diagnosis_Type', 'Residence_Area', 'Sex']
    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    df = df.dropna()

    return df


@st.cache_resource
def train_model(df):
    # Risk_Score is derived from symptoms, so it is excluded to avoid leakage.
    X = df.drop(columns=['Target', 'Risk_Score'], errors='ignore')
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion='entropy', max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'sensitivity': tp / (tp + fn),   # malaria cases correctly flagged
        'specificity': tn / (tn + fp),   # non-cases correctly cleared
        'precision': tp / (tp + fp),
        'cm': cm,
        'report': classification_report(y_test, y_pred, output_dict=True),
    }

    return clf, X, metrics


def predict_patient(clf, feature_columns, patient: dict):
    """Return (prediction, malaria probability) for one patient dict."""
    row = pd.DataFrame([patient])[feature_columns]
    prediction = int(clf.predict(row)[0])
    proba = float(clf.predict_proba(row)[0][1])
    return prediction, proba


# --------------------------------------------------------------- shared parts


def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)


def risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%'},
        title={'text': "Model-estimated likelihood of malaria"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#143d38"},
            'steps': [
                {'range': [0, 30], 'color': "#bfe5c8"},
                {'range': [30, 60], 'color': "#ffe9a8"},
                {'range': [60, 100], 'color': "#f5b7ae"}],
        }))
    fig.update_layout(height=320, margin=dict(t=60, b=10))
    return fig


# ----------------------------------------------------------------- app pages


def page_home(df, metrics):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">A screening companion, not a doctor</h2>',
                    unsafe_allow_html=True)
        st.write("""
        MalariaCheck helps you decide **how urgently to seek malaria testing**
        based on the symptoms you have right now. It compares your symptoms
        against patterns learned from real hospital records of confirmed
        malaria and non-malaria cases.

        **What it does:**
        - Estimates your malaria likelihood from 11 common symptoms and your age
        - Flags danger signs that need urgent, in-person medical care
        - Gives clear next steps — when and where to get tested

        **What it does not do:**
        - It cannot confirm or rule out malaria — only a blood test can
        - It does not store or transmit anything you enter
        """)

        st.markdown('<h3 class="sub-header">System at a glance</h3>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Patient records", f"{len(df):,}")
        with c2:
            metric_card("Confirmed malaria cases", f"{int(df['Target'].sum()):,}")
        with c3:
            metric_card("Test accuracy", f"{metrics['accuracy']:.1%}")
        with c4:
            metric_card("Sensitivity", f"{metrics['sensitivity']:.1%}")
        st.caption("Sensitivity = share of real malaria cases the model "
                   "correctly flags. Full metrics are on the Model Performance page.")

    with col2:
        st.image("malaria.jpg",
                 caption="Malaria is transmitted by infected female Anopheles mosquitoes",
                 width='stretch')

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


def page_self_assessment(clf, feature_columns):
    st.markdown('<h2 class="sub-header">Symptom Self-Assessment</h2>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Answer honestly about the symptoms you have <strong>right now or in the
        last few days</strong>. The assessment takes under a minute.
    </div>
    """, unsafe_allow_html=True)

    manual_tab, examples_tab = st.tabs(["🧍 My assessment", "📋 Example cases"])

    with manual_tab:
        with st.columns(2)[0]:
            age = st.slider("Your age", min_value=1, max_value=100, value=30,
                            key="age")

        st.markdown("#### Which of these do you have?")
        answers = {}
        cols = st.columns(2)
        for i, (feature, (label, help_text)) in enumerate(SYMPTOMS.items()):
            with cols[i % 2]:
                answers[feature] = st.checkbox(label, help=help_text,
                                               key=f"sym_{feature}")

        st.markdown('<p class="privacy-note">🔒 Nothing you enter is saved or '
                    'sent anywhere — it is processed only in this session.</p>',
                    unsafe_allow_html=True)

        if st.button("Assess my risk", type="primary", key="assess_btn"):
            patient = {'Age': age}
            patient.update({f: int(v) for f, v in answers.items()})
            prediction, proba = predict_patient(clf, feature_columns, patient)
            symptom_count = sum(patient[f] for f in SYMPTOMS)
            danger = [DANGER_SIGNS[f] for f in DANGER_SIGNS if answers.get(f)]

            st.markdown("## Your result")

            if danger:
                st.markdown(f"""
                <div class="danger-banner">
                    🚨 <strong>Danger sign reported:</strong> {', '.join(danger)}.
                    These can indicate <strong>severe malaria</strong> or another
                    serious illness. Go to the nearest health facility
                    <strong>now</strong>, regardless of the result below.
                    In Rwanda, call <strong>912</strong> for a medical emergency.
                </div>
                """, unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h3>⚠️ Your symptoms are consistent with malaria</h3>
                    <p>Estimated likelihood: <strong>{proba:.0%}</strong> ·
                    You reported <strong>{symptom_count} of 11</strong> symptoms.</p>
                    <p><strong>Get a malaria test within 24 hours.</strong>
                    Visit your nearest health centre or community health worker
                    for a rapid diagnostic test (RDT). Testing is quick and, in
                    most public facilities, free or low-cost.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h3>✅ Your symptoms are less consistent with malaria</h3>
                    <p>Estimated likelihood: <strong>{proba:.0%}</strong> ·
                    You reported <strong>{symptom_count} of 11</strong> symptoms.</p>
                    <p>Keep monitoring how you feel. If symptoms persist for
                    more than 48 hours, get worse, or new symptoms appear,
                    visit a health facility.</p>
                </div>
                """, unsafe_allow_html=True)

                if answers.get('Fever'):
                    st.warning("**You reported fever.** In a malaria-endemic "
                               "area, any fever should be tested for malaria "
                               "within 24 hours even when other symptoms are "
                               "absent — early treatment prevents severe illness.")

            st.plotly_chart(risk_gauge(proba), width='stretch')

            st.markdown("#### Recommended next steps")
            st.markdown("""
            1. **Test, don't guess** — ask for a malaria rapid diagnostic test
               (RDT) or microscopy at a health centre, clinic or pharmacy.
            2. **Don't self-medicate** with antimalarials before testing;
               incorrect treatment can mask the illness and fuel drug resistance.
            3. **Drink fluids and rest** while you arrange testing.
            4. **Return immediately** if you develop confusion, convulsions,
               difficulty breathing, dark urine, yellow eyes, or cannot keep
               fluids down.
            """)

            st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

    with examples_tab:
        st.write("Pre-filled cases that show how the model responds to "
                 "different symptom profiles.")

        example_patients = pd.DataFrame([
            {'Age': 28, 'Fever': 1, 'Headache': 1, 'Abdominal_Pain': 1,
             'General_Body_Malaise': 1, 'Dizziness': 1, 'Vomiting': 1,
             'Confusion': 0, 'Backache': 1, 'Chest_Pain': 0, 'Coughing': 0,
             'Joint_Pain': 1},
            {'Age': 45, 'Fever': 0, 'Headache': 0, 'Abdominal_Pain': 0,
             'General_Body_Malaise': 0, 'Dizziness': 0, 'Vomiting': 0,
             'Confusion': 0, 'Backache': 0, 'Chest_Pain': 0, 'Coughing': 1,
             'Joint_Pain': 0},
            {'Age': 33, 'Fever': 1, 'Headache': 1, 'Abdominal_Pain': 0,
             'General_Body_Malaise': 1, 'Dizziness': 1, 'Vomiting': 1,
             'Confusion': 1, 'Backache': 1, 'Chest_Pain': 0, 'Coughing': 0,
             'Joint_Pain': 1},
            {'Age': 60, 'Fever': 0, 'Headache': 0, 'Abdominal_Pain': 0,
             'General_Body_Malaise': 0, 'Dizziness': 0, 'Vomiting': 0,
             'Confusion': 0, 'Backache': 0, 'Chest_Pain': 0, 'Coughing': 1,
             'Joint_Pain': 0},
        ])

        X_examples = example_patients[feature_columns]
        predictions = clf.predict(X_examples)
        probabilities = clf.predict_proba(X_examples)[:, 1]
        symptom_counts = example_patients[list(SYMPTOMS)].sum(axis=1)

        results_df = pd.DataFrame({
            "Patient": [f"Patient {i + 1}" for i in range(len(predictions))],
            "Age": example_patients['Age'],
            "Symptoms reported": symptom_counts,
            "Assessment": ["Consistent with malaria" if p == 1
                           else "Less consistent" for p in predictions],
            "Estimated likelihood": [f"{p:.0%}" for p in probabilities],
        })
        st.dataframe(results_df, width='stretch', hide_index=True)

        fig = px.scatter(
            results_df.assign(Likelihood=probabilities * 100),
            x="Age", y="Likelihood", color="Assessment",
            size="Symptoms reported", hover_data=["Patient"],
            labels={"Likelihood": "Estimated likelihood (%)"},
            color_discrete_map={"Consistent with malaria": "#c0392b",
                                "Less consistent": "#1e8449"})
        fig.update_layout(yaxis_range=[-5, 105])
        st.plotly_chart(fig, width='stretch')


def page_data_insights(df):
    st.markdown('<h2 class="sub-header">Dataset Insights</h2>',
                unsafe_allow_html=True)
    st.write("The model learns from 1,622 anonymised hospital admission "
             "records with laboratory-confirmed outcomes.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), width='stretch')

    # Human-readable diagnosis label used across the charts below.
    plot_df = df.assign(
        Diagnosis=df['Target'].map({0: 'No malaria', 1: 'Malaria'}))
    diagnosis_colors = {'No malaria': '#5b9bd5', 'Malaria': '#c0392b'}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnosis distribution")
        target_counts = df['Target'].value_counts().sort_index()
        fig = px.pie(values=target_counts.values,
                     names=['No malaria', 'Malaria'],
                     color=['No malaria', 'Malaria'],
                     color_discrete_map=diagnosis_colors)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Age distribution by diagnosis")
        fig = px.histogram(plot_df, x='Age', color='Diagnosis',
                           barmode='overlay', nbins=25,
                           color_discrete_map=diagnosis_colors)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Symptom prevalence: malaria vs non-malaria patients")
    symptom_cols = list(SYMPTOMS)
    prevalence = (plot_df.groupby('Diagnosis')[symptom_cols].mean().T * 100)
    prevalence = prevalence.rename(index={c: SYMPTOMS[c][0] for c in symptom_cols})
    prevalence = prevalence.sort_values('Malaria')
    prev_long = (prevalence.reset_index()
                 .melt(id_vars='index', var_name='Diagnosis',
                       value_name='Prevalence (%)')
                 .rename(columns={'index': 'Symptom'}))

    fig = px.bar(prev_long, x='Prevalence (%)', y='Symptom', color='Diagnosis',
                 orientation='h', barmode='group',
                 color_discrete_map=diagnosis_colors)
    fig.update_layout(height=550)
    st.plotly_chart(fig, width='stretch')
    st.caption("Every symptom is more common among confirmed malaria patients, "
               "but no single symptom is decisive — which is why the model "
               "weighs combinations of symptoms.")


def page_model_performance(X, metrics, clf):
    st.markdown('<h2 class="sub-header">Model Performance</h2>',
                unsafe_allow_html=True)
    st.write("Evaluated on a held-out 30% test set the model never saw "
             "during training.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", f"{metrics['accuracy']:.1%}")
    with c2:
        metric_card("Sensitivity (recall, malaria)",
                    f"{metrics['sensitivity']:.1%}")
    with c3:
        metric_card("Specificity (recall, no malaria)",
                    f"{metrics['specificity']:.1%}")
    with c4:
        metric_card("Precision (malaria)", f"{metrics['precision']:.1%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion matrix")
        fig = px.imshow(metrics['cm'], text_auto=True, aspect="auto",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No malaria', 'Malaria'],
                        y=['No malaria', 'Malaria'],
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        <div class="info-box">
            For a screening tool, <strong>sensitivity matters most</strong>:
            missing a real malaria case (a false negative) is more dangerous
            than sending a healthy person for a test (a false positive).
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Classification report")
        report_df = pd.DataFrame(metrics['report']).transpose().round(2)
        st.dataframe(report_df, width='stretch')

        st.subheader("Feature importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=True)

        fig = px.bar(feature_importance, x='importance', y='feature',
                     orientation='h', color='importance',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, width='stretch')


def page_decision_tree(clf, X):
    st.markdown('<h2 class="sub-header">Decision Tree Visualization</h2>',
                unsafe_allow_html=True)

    st.subheader("Tree overview")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Feature importance', 'Tree structure'),
        specs=[[{"type": "bar"}, {"type": "table"}]]
    )
    fig.add_trace(
        go.Bar(x=feature_importance['importance'],
               y=feature_importance['feature'],
               orientation='h', marker_color='#14919b',
               name='Feature importance'),
        row=1, col=1)
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"],
                        fill_color='#e3e8e8', align='left'),
            cells=dict(values=[
                ["Tree depth", "Number of leaves", "Number of features",
                 "Split criterion"],
                [clf.get_depth(), clf.get_n_leaves(), len(X.columns),
                 "Entropy"]],
                fill_color='white', align='left')),
        row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Full decision tree")
    show_tree = st.radio("Display:", ["Tree overview only", "Full tree"],
                         horizontal=True, key="tree_display",
                         help="The full tree is large and takes a moment to render")

    if show_tree == "Full tree":
        with st.spinner('Rendering decision tree...'):
            tree_fig, ax = plt.subplots(figsize=(24, 12))
            plot_tree(clf, feature_names=X.columns,
                      class_names=['No malaria', 'Malaria'],
                      filled=True, rounded=True, precision=2,
                      fontsize=10, proportion=True, ax=ax)
            ax.set_title('Decision tree for malaria classification',
                         fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(tree_fig, width='stretch')

    st.subheader("How to read the tree")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Tree elements:**
        - **Nodes** — decision points based on symptoms
        - **Branches** — outcomes of each decision (yes / no)
        - **Leaves** — final assessments (malaria / no malaria)
        - **Colours** — intensity indicates confidence
        """)
    with col2:
        st.markdown("""
        **Interpretation tips:**
        - Follow the path from top to bottom
        - Each split tests one symptom or an age threshold
        - Darker colours mean higher confidence
        - The value pair shows the share of [no malaria, malaria] cases
        """)


def page_about():
    st.markdown('<h2 class="sub-header">Help & About</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        ## About MalariaCheck
        MalariaCheck is a symptom-based screening aid built with a Decision
        Tree classifier. It estimates the likelihood of malaria from your age
        and 11 common symptoms, and tells you how urgently to seek testing.

        ## How it works
        1. The model was trained on 1,622 anonymised hospital admission
           records with confirmed outcomes
        2. It learns which **combinations** of symptoms best separate malaria
           from other febrile illnesses
        3. Your answers are run through the same model to produce a likelihood
           estimate and a recommendation

        ## When to seek care immediately
        Go to a health facility **now** (in Rwanda call **912**) if you or
        someone with you has:
        - Confusion, unusual drowsiness or fainting
        - Convulsions (fits)
        - Difficulty breathing
        - Inability to keep fluids down
        - Dark or bloody urine, yellow eyes or skin
        - A sick infant, pregnant woman or elderly person with fever

        ## Limitations — please read
        - Malaria symptoms overlap with flu, typhoid and other illnesses; only
          a **blood test** can confirm malaria
        - The training data comes from one region's hospital records, so
          accuracy may differ for other populations
        - The tool does not account for pregnancy, chronic illness, or
          medication — these need professional judgement

        ## Model details
        - Algorithm: Decision Tree classifier (entropy criterion, max depth 6)
        - Training / test split: 70% / 30%, stratified by outcome
        - Features: age plus 11 reported symptoms
        - No personal data is collected, stored or transmitted

        ## Prevention
        Sleep under an insecticide-treated mosquito net, use repellents,
        remove standing water around the home, and seek testing early when
        fever appears. Pregnant women and children under five are most at risk.
        """)

    with col2:
        st.image("malaria.jpg",
                 caption="Prevention and early testing save lives",
                 width='stretch')

        st.info("""
        **Malaria facts**
        - Caused by *Plasmodium* parasites, spread by infected female
          *Anopheles* mosquitoes
        - Symptoms typically appear 7–30 days after an infective bite
        - *P. falciparum* is the most dangerous species and the most common
          in sub-Saharan Africa
        - Early diagnosis and treatment prevent nearly all deaths
        """)

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ----------------------------------------------------------------------- main


def main():
    st.markdown("""
    <div class="app-header">
        <h1>🩺 MalariaCheck</h1>
        <p>Symptom-based malaria risk assessment — know when to get tested</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner('Loading data and preparing the model...'):
        df = load_data()
        clf, X, metrics = train_model(df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES, key="nav")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚕️ MalariaCheck is a screening aid, not a diagnosis. "
        "Only a laboratory test can confirm malaria.")

    if page == "Home":
        page_home(df, metrics)
    elif page == "Self-Assessment":
        page_self_assessment(clf, list(X.columns))
    elif page == "Data Insights":
        page_data_insights(df)
    elif page == "Model Performance":
        page_model_performance(X, metrics, clf)
    elif page == "Decision Tree":
        page_decision_tree(clf, X)
    else:
        page_about()


if __name__ == "__main__":
    main()
