# # data analysis and wrangling
import pandas as pd

import feature_engineering as ft
import models

# Acquire data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

# Feature Engineering
feature_engineer = ft.FeatureEngineer()
combine = feature_engineer.engineer(combine)

# Modelisation
modeler = models.Model(combine)
prediction = modeler.main()

# Output formatting
submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": prediction
    })
submission.to_csv('output/submission.csv', index=False)