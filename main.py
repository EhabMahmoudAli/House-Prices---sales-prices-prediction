import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def preprocess_data(df):
    df['LotFrontage'].fillna(value=df['LotFrontage'].median(), inplace=True)        # median because normal distribution
    df.drop(columns=['Alley'], inplace=True)                                        # too many nulls
    df['BsmtQual'].fillna(value='TA', inplace=True)                                 # mode
    df['BsmtCond'].fillna(value='TA', inplace=True)                                 # mode
    df['BsmtExposure'].fillna(value='No', inplace=True)                             # mode
    df['BsmtFinType1'].fillna(value='Unf', inplace=True)                            # mode
    df['BsmtFinType2'].fillna(value='Unf', inplace=True)                            # mode
    df['Electrical'].fillna(value='SBrkr', inplace=True)                            # mode
    df.drop(columns=['FireplaceQu'], inplace=True)                                  # too many nulls
    df['GarageType'].fillna(value='Attchd', inplace=True)                           # mode
    df['GarageYrBlt'].fillna(value=df['GarageYrBlt'].mean(), inplace=True)          # mean because it is not normally distributed
    df['GarageFinish'].fillna(value='Unf', inplace=True)                            # mode
    df['GarageQual'].fillna(value='TA', inplace=True)                               # mode
    df['GarageCond'].fillna(value='TA', inplace=True)                               # mode
    df.drop(columns=['PoolQC', 'Fence', 'MiscFeature'], inplace=True)               # too many nulls

    # removing outliers in LotArea column
    Q1 = df['LotArea'].quantile(0.25)
    Q3 = df['LotArea'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    df.loc[(df['LotArea'] < lower_limit) | (df['LotArea'] > upper_limit), 'LotArea'] = np.nan
    df['LotArea'].fillna(value=df['LotArea'].mean(), inplace=True)

    # encoding categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df


data = preprocess_data(data)
test = preprocess_data(test)

X = data.drop(columns=['SalePrice', 'Id'])
y = np.log1p(data['SalePrice'])

# print(f'NaN values in X: {X.isnull().sum().sum()}')
# print(f'NaN values in y: {y.isnull().sum()}')

if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)  # fill with mean
if y.isnull().sum() > 0:
    y.fillna(y.mean(), inplace=True)  # fill with mean

# data splitting
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# print(f'NaN values in X_train after scaling: {np.isnan(X_train).sum()}')
# print(f'NaN values in X_valid after scaling: {np.isnan(X_valid).sum()}')

# training
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# prediction
rf_valid_preds = rf_model.predict(X_valid)
gb_valid_preds = gb_model.predict(X_valid)

# mse
rf_mse = mean_squared_error(y_valid, rf_valid_preds)
gb_mse = mean_squared_error(y_valid, gb_valid_preds)

print(f'Random Forest MSE: {rf_mse}')
print(f'Gradient Boosting MSE: {gb_mse}')

X_test = test.drop(columns=['Id'])
X_test = scaler.transform(X_test)

if np.isnan(X_test).sum() > 0:
    # print("NaN values found in X_test, handling them...")
    X_test = np.nan_to_num(X_test)  # replace nan with 0

# predictions
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# average of both predictions
final_preds = np.expm1((rf_preds + gb_preds) / 2)  # inverse log transformation

submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = final_preds

submission.to_csv('submission.csv', index=False)
