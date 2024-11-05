import os
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from matminer.featurizers.site import GaussianSymmFunc, AGNIFingerprints


sites = {19: [6.7500, 4.5000, 4.5000], 8: [4.5000, 4.5000, 6.7500], 4: [2.2500, 4.5000, 4.5000], 15: [4.5000, 4.5000, 2.2500],
         16: [4.5000, 6.7500, 4.5000], 7: [4.5000, 2.2500, 4.5000], 5: [2.2500, 6.7500, 6.7500], 13: [2.2500, 6.7500, 2.2500],
         20: [6.7500, 6.7500, 6.7500], 26: [6.7500, 6.7500, 2.2500], 0: [2.2500, 2.2500, 6.7500], 3: [2.2500, 2.2500, 2.2500],
         10: [6.7500, 2.2500, 6.7500], 18: [6.7500, 2.2500, 2.2500]}
structure_base_path = os.path.join('..', 'data', 'structures')
directories = {
    '2-metal': ['2-metal-sqs1', '2-metal-sqs2'],
    '3-metal': ['3-metal-sqs1', '3-metal-sqs2', '3-metal-sqs3'],
    '4-metal': ['4-metal-sqs']
}
files = []
for metal_type, subdirs in directories.items():
    for subdir in subdirs:
        structure_path = os.path.join(structure_base_path, subdir)
        for f in os.listdir(structure_path):
            # 将文件路径加入列表
            files.append(os.path.join('..', 'data', 'structures', subdir, f))


def get_index(site_list, file_list):
    systems = []
    for f in file_list:
        systems.append(os.path.split(f)[-1].split('.', 1)[0])
    index = []
    for i in range(len(systems)):
        index.extend([systems[i]] * len(site_list))
    return index


def get_composition(site_list, file_list):
    composition_list = []
    for f in file_list:
        structure = Structure.from_file(f)
        symbol = list(structure.symbol_set)
        index = {0: 'Ti', 1: 'Zr', 2: 'Hf', 3: 'Nb'}
        compositions = [2, 2, 2, 2]
        for i in range(4):
            if index[i] in symbol:
                compositions[i] = 1
            else:
                compositions[i] = 0
        composition_list.append(compositions)

    feature_comp = []
    for i in range(len(composition_list)):
        feature_comp.extend([composition_list[i]] * len(site_list))
    feature_comp = pd.DataFrame(feature_comp, columns=['Ti', 'Zr', 'Hf', 'Nb'])
    return feature_comp


def get_neighbor(site_list, file_list):
    feature_1NN = []
    feature_3NN = []
    feature_5NN = []
    feature_7NN = []
    for f in file_list:
        structure = Structure.from_file(f)
        for i in site_list:
            neighbor = structure.get_neighbors(structure.sites[i], 2.25)
            speices = [i.specie.name for i in neighbor]
            feature = [speices.count(i) for i in ['Ti', 'Zr', 'Hf', 'Nb']]
            feature_1NN.append(feature)
            neighbor = structure.get_neighbors_in_shell(structure[i].coords, 3.8, 0.2)
            speices = [i.specie.name for i in neighbor]
            feature = [speices.count(i) for i in ['Ti', 'Zr', 'Hf', 'Nb']]
            feature_3NN.append(feature)
            neighbor = structure.get_neighbors_in_shell(structure[i].coords, 5.03, 0.1)
            speices = [i.specie.name for i in neighbor]
            feature = [speices.count(i) for i in ['Ti', 'Zr', 'Hf', 'Nb']]
            feature_5NN.append(feature)


    feature_1NN = pd.DataFrame(feature_1NN, columns=['Ti_1NN', 'Zr_1NN', 'Hf_1NN', 'Nb_1NN'])
    feature_3NN = pd.DataFrame(feature_3NN, columns=['Ti_3NN', 'Zr_3NN', 'Hf_3NN', 'Nb_3NN'])
    feature_5NN = pd.DataFrame(feature_5NN, columns=['Ti_5NN', 'Zr_5NN', 'Hf_5NN', 'Nb_5NN'])
    return feature_1NN, feature_3NN, feature_5NN


def get_local_difference(neighbor_list):
    property_list = ("atomic number", "atomic weight", "period number", "group number", "electronegativity", "covalent radius", 'lattice constant',
                     'melting point', 'VEC', 'vacancy formation energy')
    properties = {'atomic number': [22, 40, 72, 41],
                  'atomic weight': [47.867, 91.224, 178.49, 92.90638],
                  'period number': [4, 5, 6, 5],
                  'group number': [4, 4, 4, 5],
                  'electronegativity': [1.54, 1.33, 1.3, 1.6],
                  'covalent radius': [160, 175, 175, 164],
                  'lattice constant': [4.336, 4.724, 4.650, 4.506],
                  'melting point': [3067, 3420, 3928, 3600],
                  'VEC': [8, 8, 8, 9],
                  'vacancy formation energy': [0.5915, 0.8849, 1.1883, -0.3667]}
    feature = []
    for index, row in neighbor_list.iterrows():
        neighbors = row
        neighbors = list(neighbors)
        total_neighbor = np.sum(neighbors)
        output = np.zeros((len(property_list),))
        output_dev = np.zeros((len(property_list),))
        for i, p in enumerate(property_list):
            n_props = properties[p]
            prop_mean = np.dot(neighbors, n_props) / total_neighbor
            prop_dev = np.dot(neighbors, [(x-prop_mean)**2 for x in n_props]) / total_neighbor
            output[i] = prop_mean
            output_dev[i] = prop_dev
            output_sum = np.hstack([output, output_dev])
            output_sum = list(output_sum)
        feature.append(output_sum)
    feature_sum = pd.DataFrame(feature,
                               columns=["Mean of " + p for p in property_list]
                               + ["Deviation of " + p for p in property_list])
    return feature_sum


def get_AGNI(site_list, file_list):
    AGNI = AGNIFingerprints(cutoff=5, directions=([None]))
    colnames = AGNI._generate_column_labels(multiindex=False, return_errors=False)
    feature = []
    for f in file_list:
        structure_Ti = Structure.from_file(f)
        structure_Ti.remove_species(['Zr', 'Hf', 'Nb'])
        structure_Zr = Structure.from_file(f)
        structure_Zr.remove_species(['Ti', 'Hf', 'Nb'])
        structure_Hf = Structure.from_file(f)
        structure_Hf.remove_species(['Ti', 'Zr', 'Nb'])
        structure_Nb = Structure.from_file(f)
        structure_Nb.remove_species(['Ti', 'Zr', 'Hf'])
        for s in site_list:
            feat_Ti = AGNI.featurize(structure_Ti, idx=s)
            feat_Zr = AGNI.featurize(structure_Zr, idx=s)
            feat_Hf = AGNI.featurize(structure_Hf, idx=s)
            feat_Nb = AGNI.featurize(structure_Nb, idx=s)
            feat = list(np.hstack([feat_Ti, feat_Zr, feat_Hf, feat_Nb]))
            feature.append(feat)
    feature_sum = pd.DataFrame(feature,
                               columns=['Ti_' + p for p in colnames]
                                        + ['Zr_' + p for p in colnames]
                                        + ['Hf_' + p for p in colnames]
                                        + ['Nb_' + p for p in colnames])
    return feature_sum


def get_GSF(site_list, file_list):
    GSF = GaussianSymmFunc(cutoff=5)
    colnames = GSF._generate_column_labels(multiindex=False, return_errors=False)
    feature = []
    for f in file_list:
        structure_Ti = Structure.from_file(f)
        structure_Ti.remove_species(['Zr', 'Hf', 'Nb'])
        structure_Zr = Structure.from_file(f)
        structure_Zr.remove_species(['Ti', 'Hf', 'Nb'])
        structure_Hf = Structure.from_file(f)
        structure_Hf.remove_species(['Ti', 'Zr', 'Nb'])
        structure_Nb = Structure.from_file(f)
        structure_Nb.remove_species(['Ti', 'Zr', 'Hf'])
        for s in site_list:
            feat_Ti = GSF.featurize(structure_Ti, idx=s)
            feat_Zr = GSF.featurize(structure_Zr, idx=s)
            feat_Hf = GSF.featurize(structure_Hf, idx=s)
            feat_Nb = GSF.featurize(structure_Nb, idx=s)
            feat = list(np.hstack([feat_Ti, feat_Zr, feat_Hf, feat_Nb]))
            feature.append(feat)
    feature_sum = pd.DataFrame(feature,
                               columns=['Ti_' + p for p in colnames]
                                        + ['Zr_' + p for p in colnames]
                                        + ['Hf_' + p for p in colnames]
                                        + ['Nb_' + p for p in colnames])
    return feature_sum


index = get_index(sites, files)
compositions = get_composition(sites, files)
neighbor_1NN = get_neighbor(sites, files)[0]
neighbor_3NN = get_neighbor(sites, files)[1]
neighbor_5NN = get_neighbor(sites, files)[2]
local_1NN = get_local_difference(neighbor_1NN)
local_3NN = get_local_difference(neighbor_3NN)
local_5NN = get_local_difference(neighbor_5NN)
local_1NN.columns = ['{}_1NN'.format(x) for x in list(local_1NN.columns)]
local_3NN.columns = ['{}_3NN'.format(x) for x in list(local_3NN.columns)]
local_5NN.columns = ['{}_5NN'.format(x) for x in list(local_5NN.columns)]
AGNI = get_AGNI(sites, files)
GSF = get_GSF(sites, files)
data_vac = pd.read_excel('formation_energy.xlsx')
energy = data_vac.iloc[:, -1]

feature = pd.concat([compositions, neighbor_1NN, neighbor_3NN, neighbor_5NN,
                     local_1NN, local_3NN, local_5NN,
                     AGNI, GSF, energy], axis=1)
feature.index = index
feature.to_excel('vacancy_features.xlsx', index=index)
