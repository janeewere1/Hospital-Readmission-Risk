import os
import pandas as pd
import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# this loads the test/ train data from outputs
def load_data():

    print("Currently loading, training and testing the data...")

    x_train = pd.read_csv("Outputs/X_train.csv")
    x_test = pd.read_csv("Outputs/X_test.csv")
    y_train = pd.read_csv("Outputs/y_train.csv").values.ravel()
    y_test = pd.read_csv("Outputs/y_test.csv").values.ravel()

    return x_train, x_test, y_train, y_test

# this next part is used for the training of models with the data
def model_train(x_train, y_train):

    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=500)
    lr.fit(x_train, y_train)

    print("Training LightGBM model...")
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(x_train, y_train)

    print("Training Random Forest model...")
    rfc = RandomForestClassifier(n_estimators=200, random_state=42)
    rfc.fit(x_train, y_train)

    return lr, lgbm, rfc

# this handles the evaluation part giving the accuracy score
def evaluation(model, model_name, x_test, y_test):

    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    f1 = f1_score(y_test, prediction, average='weighted')

    # this is total samples
    report = classification_report(y_test, prediction, output_dict=True)
    support = sum([v['support'] for k, v in report.items() if isinstance(v, dict)])

    print(f"\n{model_name} Performance:")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"Support  : {support}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Support": support
    }

def model_save(models):
    model_dir = "Outputs/Models"
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, f"{model_dir}/{name}.joblib")
        print(f"Saved {name} model.")


# this saves reuslts into models
def results_save(results):
    df = pd.DataFrame(results)
    df.to_csv("Outputs/Models/model_results.csv", index=False)
    print("Saved model results to Model folder in Outputs")


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    lr, rfc, lgbm = model_train(x_train, y_train)

    results = []
    results.append(evaluation(lr, "Logistic Regression", x_test, y_test))
    results.append(evaluation(lgbm, "LightGBM", x_test, y_test))
    results.append(evaluation(rfc, "Random Forest Classifier", x_test, y_test))

    results_save(results)

    model_save({
        "logistic_regression": lr,
        "lightgbm": lgbm,
        "random_forest_classifier": rfc 
    })
    print("\n Model training has been completed succesfully")

    joblib.dump(x_train.columns.tolist(), "Outputs/Models/feature_columns.joblib")   
    print("Feature columns Saved") 


def predictions(model, input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    return prediction[0], probability[0]

best_model = rfc
example_input = x_test.iloc[[0]]
pred, prob = predictions(best_model, example_input)

label_map = {
    0: "No readmission",
    1: "Readmitted after 30 days",
    2: "Readmitted within 30 days"
} 

predicted_label = label_map[pred]
confidence = prob[pred]
confidence_percent = confidence * 100 

print("\n Example Prediction Based on Dataset")
print(f"Predicted Class: {label_map[pred]}")

print("\nProbability Breakdown:")
print(f"- No Readmission: {prob[0]:.3f}")
print(f"- Readmitted >30 days: {prob[1]:.3f}")
print(f"- Readmitted <30 days: {prob[2]:.3f}")

if confidence_percent >= 70:
    risk_word = "VERY LIKELY"
elif confidence_percent >= 40:
    risk_word = "LIKELY"
else:
    risk_word = "UNLIKELY"

explanation = (
        f"This patient is {risk_word} to be {predicted_label} "
        f"({confidence_percent:.1f}% probability)."
    )
