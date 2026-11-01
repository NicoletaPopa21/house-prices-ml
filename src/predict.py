import pandas as pd
from src.preprocessing import prepare_full_data
from src.models import build_random_forest

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

X,y, preprocessor = prepare_full_data(train_data)
X_test = test_data.drop(columns = ["Id"])

final_model = build_random_forest(preprocessor)
final_model.fit(X, y)

test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
print(submission.head())