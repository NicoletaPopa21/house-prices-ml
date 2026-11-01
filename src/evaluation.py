from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def evaluate_model(y_true, y_valid):
    mae = mean_absolute_error(y_true, y_valid)
    rmse = root_mean_squared_error(y_true, y_valid)
    r2 = r2_score(y_true, y_valid)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "r2": r2
    }

