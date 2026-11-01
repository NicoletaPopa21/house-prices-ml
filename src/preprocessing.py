
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def prepare_data(data):

    # 1. Separăm caracteristicile de variabila target
    X = data.drop(columns=["SalePrice", "Id"])
    y = data["SalePrice"]

    # 2. Împărțim datele în train și validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Identificăm coloanele numerice și categorice
    numerical_cols = X_train.select_dtypes(include=["number"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    # 4. Preprocesarea coloanelor numerice
    numerical_transformer = SimpleImputer(strategy="median")

    # 5. Preprocesarea coloanelor categorice
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 6. Combinăm transformările
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 7. Returnăm datele și preprocesorul
    return X_train, X_valid, y_train, y_valid, preprocessor
