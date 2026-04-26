import os
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# this loads the model results
def result_loading():
    results_path = "Outputs/Models/model_results.csv"
    print("Loading results from:", results_path, "...")

    if not os.path.exists(results_path):
        raise FileNotFoundError("model_results.csv not found. run model-training.py first to proceed.")
    return pd.read_csv(results_path)

# this creates and visualises the bar chart
def accuracy_bar_chart(results_df):

    models = results_df["Model"]
    accuracy_score = results_df["Accuracy"]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracy_score)

    plt.title("Model Accuracy Comparison Chart")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy Score (decimal)")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle= '--', alpha=0.6)

    output_dir = "Outputs/Visualisation"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/accuracy_comparison.png"
    plt.savefig(save_path, dpi=300)

    print("Comparison graph has been saved to Visualisation folder in Models")

    plt.show()

#this creates a metrics bar chart
def metrics_bar_chart(results_df):

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 7))

    plt.title("Model Performance Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig("Outputs/Visualisation/performance.png", dpi=300)

    plt.show()

# this is for the support chart
def support_chart(results_df):

    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Model"], results_df["Support"])

    plt.title("Support (Number of Samples Used)")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Support")

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig("Outputs/Visualisation/support_comparison.png", dpi=300)

    plt.show()

def confusion_matrix_plot(model, x_test, y_test, model_name):

    label_map = {
        0: "No readmission",
        1: "Readmitted after 30 days",
        2: "Readmitted within 30 days"
    }

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")


    plt.subplots_adjust(right=0.75)
    legend_text = "\n".join([f"{k}: {v}" for k, v in label_map.items()])
    plt.gca().text(
        1.25, 1.3, legend_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    os.makedirs("Outputs/Visualisation", exist_ok=True)
    plt.savefig(f"Outputs/Visualisation/{model_name}_confusion_matrix.png", dpi=300)

    plt.show()

#this is for shap
def shap_scatter_plot(model, x_test, class_index=2, top_n=3):
    print(f"\nGenerating SHAP scatter plots for  {top_n} features in class {class_index}...")

    label_map = {
        0: "No readmission",
        1: "Readmitted after 30 days",
        2: "Readmitted within 30 days"
    }

    output_dir = "Outputs/Visualisation/SHAP"
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(x_test)

    # this handles different SHAP output formats
    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    else:
        class_shap_values = shap_values[:, :, class_index]

    # this ranks features by mean absolute SHAP value
    mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    for i in top_feature_indices:
        feature_name = x_test.columns[i]
        print(f"Creating scatter plot for: {feature_name}")

        shap.dependence_plot(
            i,
            class_shap_values,
            x_test,
            feature_names=x_test.columns,
            show=False
        )

        plt.title(f"Random Forest - {feature_name} ({label_map[class_index]})")
        file_name = f"random_forest_{feature_name}_class_{class_index}_scatter.png".replace(" ", "_")
        plt.savefig(f"{output_dir}/{file_name}", dpi=300, bbox_inches="tight")
        plt.close()




def shap_rf_summary_plot(model, x_test, class_index=0, max_display=10):
    print(f"\nGenerating Random Forest SHAP summary plot for class {class_index}...")

    output_dir = "Outputs/Visualisation/SHAP"
    os.makedirs(output_dir, exist_ok=True)

    label_map = {
        0: "No readmission",
        1: "Readmitted after 30 days",
        2: "Readmitted within 30 days"
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    else:
        class_shap_values = shap_values[:, :, class_index]

    class_colors = {
        0: "forestgreen",
        1: "darkorange",
        2: "darkred"
    }

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        class_shap_values,
        x_test,
        plot_type="bar",
        max_display=max_display,
        color=class_colors[class_index],
        show=False
    )

    plt.title(f"Random Forest SHAP Summary Plot ({label_map[class_index]})")
    plt.savefig(
        f"{output_dir}/rf_summary_plot_class_{class_index}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

   
# this prints metrics table 
def print_metrics(results_df):
    print("\nModel Performance:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    results_df = result_loading()

    print_metrics(results_df) 

    accuracy_bar_chart(results_df)
    metrics_bar_chart(results_df)
    support_chart(results_df)

    # this loads test data
    x_test = pd.read_csv("Outputs/X_test.csv")
    y_test = pd.read_csv("Outputs/y_test.csv").values.ravel()
    x_train = pd.read_csv("Outputs/X_train.csv")

    # this loads models
    lr = joblib.load("Outputs/Models/logistic_regression.joblib")
    lgbm = joblib.load("Outputs/Models/lightgbm.joblib")
    rfc = joblib.load("Outputs/Models/random_forest_classifier.joblib")

    # this plots confusion matrices
    confusion_matrix_plot(lr, x_test, y_test, "Logistic Regression")
    confusion_matrix_plot(lgbm, x_test, y_test, "LightGBM")
    confusion_matrix_plot(rfc, x_test, y_test, "Random Forest")

    #this plots the shap values
    shap_rf_summary_plot(rfc, x_test, class_index=0, max_display=10)
    shap_rf_summary_plot(rfc, x_test, class_index=1, max_display=10)
    shap_rf_summary_plot(rfc, x_test, class_index=2, max_display=10)

    shap_scatter_plot(rfc, x_test,class_index=0, top_n=3)
    shap_scatter_plot(rfc, x_test,class_index=1, top_n=3)
    shap_scatter_plot(rfc, x_test,class_index=2, top_n=3)
   
