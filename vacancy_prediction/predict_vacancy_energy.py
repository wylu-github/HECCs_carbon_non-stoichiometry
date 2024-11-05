import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge as KRR
from get_vacancy_features import get_composition, get_neighbor, get_local_difference, get_AGNI, get_GSF
from pymatgen.core.structure import Structure

KR = KRR(kernel='poly', alpha=1)
data = pd.read_excel(r'vacancy_features.xlsx')
X = np.array(data.iloc[:, 1:-1])
y = np.array(data.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=20)

scaler = StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

KR.fit(X_train, y_train)
score_r2_train = r2_score(KR.predict(X_train), y_train)
score_mae_train = MAE(KR.predict(X_train), y_train)
score_r2 = r2_score(KR.predict(X_test), y_test)
score_mae = MAE(KR.predict(X_test), y_test)
print('the r2 of {} is: {}, {}'.format('KR', score_r2_train, score_r2),
      'the mae of {} is: {}, {}'.format('KR', score_mae_train, score_mae))


user_input_file = input("Please enter the path to the .vasp structure file for prediction: ")
structure_initial = Structure.from_file(user_input_file)
carbon_list = [i for i, s in enumerate(structure_initial.sites) if s.specie.name == 'C']
compositions = get_composition(carbon_list, [user_input_file])
neighbor_1NN = get_neighbor(carbon_list, [user_input_file])[0]
neighbor_3NN = get_neighbor(carbon_list, [user_input_file])[1]
neighbor_5NN = get_neighbor(carbon_list, [user_input_file])[2]
local_1NN = get_local_difference(neighbor_1NN)
local_3NN = get_local_difference(neighbor_3NN)
local_5NN = get_local_difference(neighbor_5NN)
AGNI = get_AGNI(carbon_list, [user_input_file])
GSF = get_GSF(carbon_list, [user_input_file])
feature = pd.concat([compositions, neighbor_1NN, neighbor_3NN, neighbor_5NN,
                     local_1NN, local_3NN, local_5NN,
                     AGNI, GSF], axis=1)

feature_scaler = scaler.transform(np.array(feature))

E_predict = KR.predict(feature_scaler).reshape(len(np.array(feature_scaler)), 1)

print("Predicted carbon vacancy formation energies (including carbon atom indices):")
for idx, (carbon_index, energy) in enumerate(zip(carbon_list, E_predict)):
    print(f"Site index: {idx}, Carbon atom index: {carbon_index}, Predicted formation energy: {energy[0]}")
