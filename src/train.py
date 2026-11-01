import pandas as pd

from src.preprocessing import prepare_data
from src.models import build_linear_regression, build_decision_tree, build_random_forest, tune_random_forest
from src.evaluation import evaluate_model

data = pd.read_csv("data/train.csv")

x_train, x_valid, y_train, y_valid, preprocessor = prepare_data(data)


#===================== LINEAR REGRESION =================
model = build_linear_regression(preprocessor)

model.fit(x_train, y_train)
predictions = model.predict(x_valid)

results = evaluate_model(y_valid, predictions)
"""print(results)

print("Linear Regression")
print(f"MAE: {results['MAE']:.2f}")
print(f"RMSE: {results['RMSE']:.2f}")"""


#================== DECISION TREE ===============================
decision_tree = build_decision_tree(preprocessor)

decision_tree.fit(x_train, y_train)
tree_predictions=decision_tree.predict(x_valid)

tree_results = evaluate_model(y_valid, tree_predictions)
"""print("Decision Tree")
print(tree_results)"""

#==================== RANDOM FOREST ==============================
random_forest = build_random_forest(preprocessor)
random_forest.fit(x_train, y_train)
random_forest_predictions = random_forest.predict(x_valid)
random_forest_result = evaluate_model(y_valid, random_forest_predictions)
"""print("Random Forest")
print(random_forest_result)"""

#
#======= saving the results

comparison = pd.DataFrame({
    "Linear Regression:": results,
    "Decision Tree:": tree_results,
    "Random Forest: ": random_forest_result
})
print("\nModel Comparison:")
print(comparison)

# ============== RANDOM FOREST OPTIMIZATION ==============

search = tune_random_forest(preprocessor, x_train, y_train)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation RMSE:")
print(-search.best_score_)

# ============== OPTIMIZED RANDOM FOREST EVALUATION ==============

best_model = search.best_estimator_

optimized_predictions = best_model.predict(x_valid)

optimized_results = evaluate_model(y_valid, optimized_predictions)

print("\nOptimized Random Forest:")
print(optimized_results)