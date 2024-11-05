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


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


LR = LinearRegression()
LASSO = Lasso(alpha=0.001, max_iter=100000)
RR = Ridge(alpha=0.5)
KR = KRR(kernel='poly', alpha=1)
KNN = KNR(n_neighbors=10)
SVM = SVR(kernel='rbf', C=10000, gamma=1e-05)
RF = RFR(n_estimators=50, max_depth=8, random_state=20)
GBoost = GBR(n_estimators=100, max_depth=3, random_state=20)
XGBoost = XGB(n_estimators=70, max_depth=3, random_state=20)
ANN = MLPRegressor(activation='identity', max_iter=100000, hidden_layer_sizes=(9, 1), random_state=20)

model = [LR, LASSO, RR, KR, SVM, RF, GBoost, XGBoost, ANN]

data = pd.read_excel(r'vacancy_features.xlsx')
X = np.array(data.iloc[:, 1:-1])
y = np.array(data.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=20)

scaler = StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

r2_train = []
r2 = []
mae_train = []
mae = []
for m in model:
    m.fit(X_train, y_train)
    score_r2_train = r2_score(m.predict(X_train), y_train)
    score_mae_train = MAE(m.predict(X_train), y_train)
    score_r2 = r2_score(m.predict(X_test), y_test)
    score_mae = MAE(m.predict(X_test), y_test)
    r2_train.append(score_r2_train)
    r2.append(score_r2)
    mae_train.append(score_mae_train)
    mae.append(score_mae)
    print('the r2 of {} is: {}, {}'.format(namestr(m, globals())[0], score_r2_train, score_r2),
          'the mae of {} is: {}, {}'.format(namestr(m, globals())[0], score_mae_train, score_mae))


