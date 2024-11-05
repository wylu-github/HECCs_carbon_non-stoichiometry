import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neighbors import KNeighborsRegressor as KNR
from get_mechanical_features import get_composition, get_neighbor, get_local_difference, get_AGNI, get_GSF
from pymatgen.core.structure import Structure

# Define your machine learning models
KR = KRR(kernel='poly', alpha=0.1)
KNN1 = KNR(n_neighbors=10)
KNN2 = KNR(n_neighbors=10)

# Load training data
data = pd.read_excel(r'features_mechanical.xlsx')
X = np.array(data.iloc[:, 1:-4])
y_G = np.array(data.iloc[:, -4])
y_B = np.array(data.iloc[:, -3])
y_E = np.array(data.iloc[:, -2])

# Split the data for training and testing
X_train, X_test, y_E_train, y_E_test, y_G_train, y_G_test, y_B_train, y_B_test = train_test_split(
    X, y_E, y_G, y_B, train_size=0.8, random_state=20)

# Scale the features
scaler = StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the models
KR.fit(X_train, y_E_train)
KNN1.fit(X_train, y_G_train)
KNN2.fit(X_train, y_B_train)


# Function to predict mechanical properties for a given structure
def predict_mechanical_properties(structure_file):
    try:
        # Load the input structure
        structure_initial = Structure.from_file(structure_file)
        carbon_list = [i for i, s in enumerate(structure_initial.sites) if s.specie.name == 'C']

        # Feature extraction
        compositions = get_composition(carbon_list, [structure_file])
        neighbor_1NN = get_neighbor(carbon_list, [structure_file])[0]
        neighbor_3NN = get_neighbor(carbon_list, [structure_file])[1]
        neighbor_5NN = get_neighbor(carbon_list, [structure_file])[2]
        local_1NN = get_local_difference(neighbor_1NN)
        local_3NN = get_local_difference(neighbor_3NN)
        local_5NN = get_local_difference(neighbor_5NN)
        AGNI = get_AGNI(carbon_list, [structure_file])
        GSF = get_GSF(carbon_list, [structure_file])

        # Concatenate features
        feature = pd.concat([compositions, neighbor_1NN, neighbor_3NN, neighbor_5NN,
                             local_1NN, local_3NN, local_5NN,
                             AGNI, GSF], axis=1)

        # Scale the extracted features
        feature_scaler = scaler.transform(np.array(feature))

        # Make predictions for the given structure
        E_predict = KR.predict(feature_scaler).reshape(len(np.array(feature_scaler)), 1)
        G_predict = KNN1.predict(feature_scaler).reshape(len(np.array(feature_scaler)), 1)
        B_predict = KNN2.predict(feature_scaler).reshape(len(np.array(feature_scaler)), 1)

        print("Predicted mechanical properties (including site and carbon atom indices):")
        for idx, (carbon_index, E, G, B) in enumerate(zip(carbon_list, E_predict, G_predict, B_predict)):
            print(f"Site index: {idx}, Carbon atom index: {carbon_index}, "
                  f"Predicted Elastic Modulus (E): {E[0]}, "
                  f"Predicted Shear Modulus (G): {G[0]}, "
                  f"Predicted Bulk Modulus (B): {B[0]}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Ask the user to input the .vasp structure file path
structure_file = input("Please input the path to the structure (.vasp) file: ")
predict_mechanical_properties(structure_file)
