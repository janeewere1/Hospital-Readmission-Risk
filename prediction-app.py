import streamlit as st
import numpy as np
import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from openai import OpenAI


# this configures the app page
st.set_page_config(
    page_title="Hospital Readmissiom Risk Calculator System",
    layout="wide"
)
st.title("Hospital Readmission Risk Predictor")
st.markdown("AI Assisted Clinical Decision support prototype")

st.warning("This tool acts as decision support vector only and does NOT replace clinical Judegment. ")

#his initialises the session state
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False


# this part loads the model and its features
model_path = "Outputs/Models/random_forest_classifier.joblib"
feature_path = "Outputs/Models/feature_columns.joblib"

model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)
explainer = shap.TreeExplainer(model)

# this is for the sidebar input panel
st.sidebar.header("Input Patient Information")

age = st.sidebar.slider("Age", 0, 100, 50)

time_in_hospital = st.sidebar.slider(
    "Length of Hospital Stay (days)", 1, 28, 3
)

number_lab_procedures = st.sidebar.slider(
    "Number of Lab Procedures", 0, 150, 40
)

number_medications = st.sidebar.slider(
    "Number of Medications", 0, 50, 10
)

number_inpatient = st.sidebar.slider(
    "Previous Inpatient Visits", 0, 20, 0
)

number_emergency = st.sidebar.slider(
    "Previous Emergency Visits", 0, 20, 0
)

number_outpatient = st.sidebar.slider(
    "Previous Outpatient Visits", 0, 20, 0
)

number_diagnoses = st.sidebar.slider(
    "Number of Diagnoses", 1, 10, 3
)
# this creates a full feature dictionary with defaults
input_data = {feature: 0 for feature in feature_columns}

# this replaces the important features with the user input
input_data.update({
    "age": age,
    "time_in_hospital": time_in_hospital,
    "number_lab_procedures": number_lab_procedures,
    "number_medications": number_medications,
    "number_inpatient": number_inpatient,
    "number_emergency": number_emergency,
    "number_outpatient": number_outpatient,
    "number_diagnoses": number_diagnoses
})

input_df = pd.DataFrame([input_data])

# this is part generates a prediction
if st.sidebar.button("Generate Prediction"):
    st.session_state.prediction_run = True

#this runs the prediction
if st.session_state.prediction_run:


    input_df = input_df[feature_columns] 

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    label_map = {
        0: "No Readmission Predicted",
        1: "Readmission Predicted After 30 Days",
        2: "Readmission Predicted Within 30 Days"
    }

    predicted_label = label_map[prediction]

    # this determines the highest probability
    max_prob = max(probability)
    risk_percentage = max_prob * 100

    # classify risk level
    if risk_percentage < 40:
        risk_level = "LOW RISK"
        risk_colour = "green"
    elif risk_percentage < 70:
        risk_level = "MODERATE RISK"
        risk_colour = "orange"
    else:
        risk_level = "HIGH RISK"
        risk_colour = "red"

    # this is the Risk Badge
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Outcome")
        st.metric("Predicted Class", predicted_label)

        st.markdown(
        f"""
        <h3 style='color:{risk_colour};'>
        Clinical Risk Level: {risk_level}
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.progress(max_prob)

    with col2:
        st.subheader("Probability Breakdown")
        st.write(f"No Readmission: {probability[0]:.2f}")
        st.write(f">30 Days: {probability[1]:.2f}")
        st.write(f"<30 Days: {probability[2]:.2f}")

    st.divider()

    # this is for the SHAP Explanation
    shap_values = explainer(input_df)

    # this extract the SHAP values for predicted class
    impact_values = shap_values.values[0][:, prediction]

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": impact_values
    })

    shap_df["Absolute Impact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values(by="Absolute Impact", ascending=False)

    top_features = shap_df.head(5)

    st.subheader("Top Contributing Factors")
    st.dataframe(top_features[["Feature", "Impact"]])

    #this is for the SHAP impact chart
    st.subheader("Feature Impact Visualisation")
    fig, ax = plt.subplots()

    ax.barh(
        top_features["Feature"],
        top_features["Impact"]
    )

    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Top Factors Influencing Readmission Risk")

    ax.invert_yaxis()

    st.pyplot(fig)

    st.divider()

    # this prepares the SHAP summary
    feature_summary = "\n".join(
        [f"{row.Feature}: contribution {row.Impact:.3f}"
        for _, row in top_features.iterrows()]
    )

    # this is for using LLM to explain
    if st.checkbox("Generate AI Clinical Explanation using LLM"):
        try:

            from openai import OpenAI

            client = OpenAI(api_key=st.secrets["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

            prompt = f"""
            A hospital readmission model predicted: {predicted_label}.

            Probability breakdown:
            No Readmission Predicted: {probability[0]:.2f}
            Readmission Predicted After 30 Days: {probability[1]:.2f}
            Readmission Predicted Within 30 Days: {probability[2]:.2f}

        
            The following SHAP feature contributions explain the model prediction: 
        
            {feature_summary}

            Provide a concise clinical explanation why these factors contributed to the predicted readmission risk.
        
            Important:
            - Emphasise uncertainty and the probabilistic nature.
            - Do not invent any additional factors.
            - Write in a professional clinical decision-support tone.
            """

            with st.spinner("Generating AI clinical interpretation..."):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )

                explanation = response.choices[0].message.content

                st.subheader("AI Clinical Interpretation Explanation")
                st.write(explanation)
            

        except Exception as e:
            st.error(f"LLM explanation could not be generated: {e}")
            
    st.divider()

    if st.checkbox("Generate AI Clinical Recomendations"):
                    
        try:
            client = OpenAI(api_key=st.secrets["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

            recommendation_prompt = f"""
            A hospital readmission prediction model has produced the following result.

            Prediction: {predicted_label}

            Probability breakdown:
            No Readmission: {probability[0]:.2f}
            Readmission >30 days: {probability[1]:.2f}
            Readmission <30 days: {probability[2]:.2f}

            Key factors influencing the prediction:
            {feature_summary}

            Based on these factors, provide general clinical management considerations
            that may help reduce hospital readmission risk.

            Important:
            - These should be high-level recommendations.
            - Do NOT provide specific medical treatment instructions.
            - Emphasise that these suggestions are only to support clinician judgement.
            """

            with st.spinner("Generating AI clinical recommendations..."):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": recommendation_prompt}
                ]
            )

            recommendations = response.choices[0].message.content

            st.subheader("AI Clinical Recommendations")
            st.write(recommendations)

        except Exception as e:
            st.error(f"AI recommendations could not be generated: {e}")



