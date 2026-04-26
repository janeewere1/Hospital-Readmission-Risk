# this imports all the necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# this section cleans the dataset by: removing any duplicates, replacing unknown values like ? with NaN and dropping any missing values.
def clean_data(data):
    print("Currently cleaning the dataset...")
    
    # this part replaces unknown values
    data = data.replace("?", pd.NA)

    # this drops columns that are not relevant 
    data = data.dropna(axis=1, thresh=0.5 * len(data))

     # this drops rows with missing values
    data = data.dropna(how="all")

    for col in data.select_dtypes(include= 'object'):
        data[col] = data[col].fillna("Unknown")

    for col in data.select_dtypes(include=['int64', 'float64']):
        data[col] = data[col].fillna(data[col].median())

    return data


# this loads and prepares the dataset, splits the dataset into train/test split and encodes categorical variables
def load_and_prepare_data(filepath):

    print(f"Loading dataset from folder ...")
    data = pd.read_csv(filepath)
    
    filepath = "C:/Users/2204805/OneDrive - University of Wolverhampton/6CS007/Hospital Readmissions project/Data/diabetic_data.csv"

    print("Initial data shape:", data.shape)
    
    # this cleans the data
    data = clean_data(data)

    # this is to save te cleaned datasets into the outpusts folder
    output_dir = "Outputs"
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(f"{output_dir}/cleaned_data_raw.csv", index=False)

    # this encodes categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    # this ensures there is readmitted column
    if 'readmitted' not in data.columns:
        raise ValueError("'readmitted' column not found in dataset.")
    x = data.drop('readmitted', axis=1)
    y = data['readmitted']

    # this is for the 80/20 test split
    x_train, x_test, y_train, y_test = train_test_split(
       x, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")

    # this saves the train/test splits
    x_train.to_csv(f"{output_dir}/x_train.csv", index=False)
    x_test.to_csv(f"{output_dir}/x_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Dataset has been cleaned and saved into '{output_dir}/'")

    return x_train, x_test, y_train, y_test

if __name__== "__main__":
    x_train, x_test, y_train, y_test = load_and_prepare_data("data/diabetic_data.csv")
    print("Data preparation has been completed successfully")
