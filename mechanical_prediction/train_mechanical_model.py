import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost.sklearn import XGBRegressor as XGB
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neural_network import MLPRegressor


# Get the name of the model
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# Load data
data = pd.read_excel(r'features_mechanical.xlsx')
X = np.array(data.iloc[:, 1:-4])
y_vars = {
    'C44': np.array(data.iloc[:, -1]),
    'G': np.array(data.iloc[:, -4]),
    'B': np.array(data.iloc[:, -3]),
    'E': np.array(data.iloc[:, -2])
}

# Define model list
model = [
    LinearRegression(),
    Lasso(alpha=0.001, max_iter=1000000),
    Ridge(alpha=0.001),
    KRR(kernel='poly', alpha=0.1),
    KNR(n_neighbors=10),
    SVR(kernel='rbf', C=100, gamma=0.001),
    RFR(n_estimators=30, max_depth=9, random_state=20),
    GBR(n_estimators=60, max_depth=5, random_state=20),
    XGB(n_estimators=100, max_depth=3, random_state=20),
    MLPRegressor(activation='identity', max_iter=100000, hidden_layer_sizes=(5, 17), random_state=20)
]


# Function to train and evaluate models
def train_and_evaluate(X, y, target_name):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=20)

    # Scale features
    scaler = StandardScaler().fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize lists to store results
    r2_train_list, r2_list, mae_train_list, mae_list = [], [], [], []

    # Iterate through each model
    for m in model:
        # Fit the model to the training data
        m.fit(X_train, y_train)

        # Predictions for train and test data
        y_train_pred = m.predict(X_train)
        y_test_pred = m.predict(X_test)

        # Calculate R2 and MAE for train and test sets
        score_r2_train = r2_score(y_train, y_train_pred)
        score_mae_train = MAE(y_train, y_train_pred)
        score_r2 = r2_score(y_test, y_test_pred)
        score_mae = MAE(y_test, y_test_pred)

        # Store results in the corresponding lists
        r2_train_list.append(score_r2_train)
        r2_list.append(score_r2)
        mae_train_list.append(score_mae_train)
        mae_list.append(score_mae)

        # Get model class name for printing
        model_name = type(m).__name__

        # Print the results for each model
        print(f'The {target_name} R2 of {model_name} is : {score_r2_train}, {score_r2}',
              f'The {target_name} MAE of {model_name} is: {score_mae_train}, {score_mae}')

    return pd.DataFrame({
        'r2_train': r2_train_list,
        'r2_test': r2_list,
        'mae_train': mae_train_list,
        'mae_test': mae_list
    })


# Loop through each target variable and evaluate models
for target_name, y in y_vars.items():
    print(f"\nTraining and evaluating for {target_name}...\n")
    accuracy = train_and_evaluate(X, y, target_name)
