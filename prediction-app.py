import streamlit as st
import numpy as np
import os
import pandas as pd
import joblib
import shap
import openai


# this configures the app page
st.set_page_config(
    page_title="Hospital Readmissiom Risk Calculator System",
    layout="wide"
)
st.title("Hospital Readmission Risk Predictor")
st.markdown("AI Augmented Clinical Decision support prototype")

st.warning("This tool acts as decision support vector only and does NOT replace clinical Judegment. ")

# this part loads the model and its features
model_path = "Outputs/Models/random_forest_classifier.joblib"
feature_path = "Outputs/Models/feature_columns.joblib"

model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)
explainer = shap.TreeExplainer(model)

# this is for the sidebar input panel
st.sidebar.header("Input Patient Information")
input_data = {}

for feature in feature_columns:
    input_data[feature] = st.sidebar.number_input(
        f"{feature}",
        value=0.0
    )
input_df = pd.DataFrame([input_data])

# this is part generates a prediction
if st.sidebar.button("Generate Prediction"):
    input_df = input_df[feature_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    label_map = {
        0: "No Readmission Predicted",
        1: "Readmission Predicted After 30 Days",
        2: "Readmission Predicted Within 30 Days"
    }

    predicted_label = label_map[prediction]

    # this is the Risk Badge
    col1, col2 = st.colums(2)
    with col1:
        st.subheader("Prediction Outcome")
        st.metric("Predicted Class", predicted_label)

    with col2:
        st.subheader("Probability Breakdown")
        st.write(f"No Readmission: {probability[0]:.2f}")
        st.write(f">30 Days: {probability[1]:.2f}")
        st.write(f"<30 Days: {probability[2]:.2f}")

    # this is for the SHAP Explanation
    shap_values = explainer.shap_values(input_df)

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": shap_values[prediction][0]
    })

    shap_df["Absolute Impact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values(by="Absolute Impact", ascending=False)

    top_features = shap_df.head(5)

    st.subheader("Top Contributing Factors")
    st.dataframe(top_features[["Feature", "Impact"]])

    # this is for using LLM to explain
    if st.checkbox("Generate AI Clinical Explanation using LLM"):

        openai.api_key = os.getenv("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")

        feature_summary = "\n".join(
            [f"- {row.Feature} (impact: {row.Impact:.3f})"
             for _, row in top_features.iterrows()]
        )

        prompt = f"""
        A hospital readmission model predicted: {predicted_label}.

        Probability breakdown:
        No Readmission Predicted: {probability[0]:.2f}
        Readmission Predicted After 30 Days: {probability[1]:.2f}
        Readmission Predicted Within 30 Days: {probability[2]:.2f}

        Key contributing features:
        {feature_summary}

        Provide a concise clinical explanation.
        Emphasise uncertainty and the probabilistic nature.
        Do not invent any additional factors.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            explanation = response['choices'][0]['message']['content']

            st.subheader("AI Clinical Explanation")
            st.write(explanation)

        except:
            st.error("LLM explanation could not be generated. Check API key.")





