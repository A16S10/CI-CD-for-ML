import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import datetime

def list_of_datasets(data_directory):
    """List all dataset files in the specified directory."""
    dataset_files = os.listdir(data_directory)
    dataset_paths = [os.path.join(data_directory, dataset_file) for dataset_file in dataset_files]
    
    print("Datasets found:")
    for dataset_file in dataset_files:
        print(dataset_file)
    
    return dataset_paths

def train_and_evaluate(dataset_path):
    """Train and evaluate the model using the specified dataset."""
    # Load the dataset
    drug_df = pd.read_csv(dataset_path)
    drug_df.head()

    # Split features and labels
    X = drug_df.drop("Drug", axis=1).values
    y = drug_df.Drug.values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

    # Specify categorical and numerical columns
    cat_col = [1, 2, 3]
    num_col = [0, 4]

    # Create preprocessing and model pipeline
    transform = ColumnTransformer(
        [
            ("encoder", OrdinalEncoder(), cat_col),
            ("num_imputer", SimpleImputer(strategy="median"), num_col),
            ("num_scaler", StandardScaler(), num_col),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocessing", transform),
            ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
        ]
    )

    # Train the model
    pipe.fit(X_train, y_train)

    # Make predictions
    predictions = pipe.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"Accuracy: {round(accuracy * 100, 2)}%, F1: {round(f1, 2)}")

    # Extract dataset name (without extension) and current timestamp
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a unique results directory based on dataset name and timestamp
    results_dir = os.path.join("Results", f"{dataset_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics to a text file
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w") as outfile:
        outfile.write(f"Accuracy = {round(accuracy * 100, 2)}%, F1 Score = {round(f1, 2)}\n")

    
    # Save the model pipeline to a file using pickle
    model_path = os.path.join("Model", f"{dataset_name}_pipeline_{timestamp}.pkl")
    os.makedirs("Model", exist_ok=True)  # Ensure Model directory exists
    with open(model_path, "wb") as model_file:
        pickle.dump(pipe, model_file)

    print(f"Results saved in: {results_dir}")
    print(f"Model saved as: {model_path}")

if __name__ == "__main__":
    # Ensure the dataset directory is passed correctly
    import sys
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        print("Error: No dataset directory provided.")
        sys.exit(1)
    
    # Step 1: List all datasets
    dataset_paths = list_of_datasets(data_directory)
    
    # Step 2: Train and evaluate each dataset
    for dataset_path in dataset_paths:
        print(f"\nProcessing dataset: {dataset_path}")
        train_and_evaluate(dataset_path)
