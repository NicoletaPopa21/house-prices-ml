from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV


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

"""
Functie ce incearca 20 de combinatii ale parametrilor
Folosim cross validation cu 3 fold uri
Alege combinatia cu cel mai bun rmse mediu 
"""
def tune_random_forest(preprocessor, X_train, y_train):

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [100,200,300,500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2,5,10],
        "model__min_samples_leaf": [1,2,4],
        "model__max_features": [1.0, "sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator = model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=3
    )

    search.fit(X_train, y_train)
    return search