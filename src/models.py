from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def build_linear_regression(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])


def build_decision_tree(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42))
    ])


def build_random_forest(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
