# data analysis and wrangling
import pandas as pd
import numpy as np


class FeatureEngineer(object):
    def engineer(self, combine):
        print("feature engineering beginning")
        train_df = combine[0]
        test_df = combine[1]

        # Remove features with more than 20% missing:
        train_df = train_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage"], axis=1)
        test_df = test_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage"], axis=1)

        # Remove "Garage" features:
        train_df = train_df.drop(["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"], axis=1)
        test_df = test_df.drop(["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"], axis=1)

        # Remove Bsmt features
        train_df = train_df.drop(["BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1"], axis=1)
        test_df = test_df.drop(["BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1"], axis=1)

        # Remove MasVnr features
        train_df = train_df.drop(["MasVnrType", "MasVnrArea"], axis=1)
        test_df = test_df.drop(["MasVnrType", "MasVnrArea"], axis=1)

        # Remove the entry with "Electrical" missing
        df_train = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)

        combine = [train_df, test_df]

        quantitative = [f for f in train_df.columns if train_df.dtypes[f] != 'object']
        quantitative.remove('SalePrice')
        quantitative.remove('Id')
        qualitative = [f for f in train_df.columns if train_df.dtypes[f] == 'object']




        # Convert categorical variable into one-hot
        train_df = pd.get_dummies(train_df)
        test_df = pd.get_dummies(test_df)

        combine = [train_df, test_df]
        print("feature engineering done")
        return combine
