import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import dagshub
import os

# Setup DagsHub + MLflow
dagshub.init(
    repo_owner='ArfaniAsra',
    repo_name='Eksperimen_SML_Muhammad-Arfani-Asra',
    mlflow=True
)

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        'healthcare-dataset-stroke-data_preprocessing.csv'
    )

    X_train, X_test, y_train, y_test = load_data(data_path)

    mlflow.set_experiment("stroke-prediction-baseline")

    with mlflow.start_run(run_name="random-forest-baseline"):
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

if __name__ == '__main__':
    main()