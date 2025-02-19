# Non-stoichiometric HECCs Prediction Repository

This repository contains scripts and models for predicting mechanical properties and carbon vacancy formation energies of high-entropy carbide ceramics.

## Directory Structure

```plaintext
├── data
│   └── structures          # Structure files for input data
│       ├── 2-metal-sqs1    # Configurations for 2-metal systems
│       ├── 2-metal-sqs2
│       ├── 3-metal-sqs1    # Configurations for 3-metal systems
│       ├── 3-metal-sqs2
│       ├── 3-metal-sqs3
│       └── 4-metal-sqs     # Configurations for 4-metal systems
├── mechanical_prediction   # Scripts and models for mechanical property prediction
│   ├── get_mechanical_features.py      # Extracts features for mechanical prediction
│   ├── predict_mechanical_properties.py # Loads models to predict properties for new data
│   └── train_mechanical_model.py       # Trains models for mechanical property prediction
├── models                  # Saved models for both mechanical and vacancy predictions
│   ├── mechanical_prediction
│   └── vacancy_prediction
├── vacancy_prediction      # Scripts and models for carbon vacancy prediction
│   ├── get_vacancy_features.py        # Extracts features for vacancy prediction
│   ├── predict_vacancy_properties.py   # Loads models to predict vacancy formation energy
│   └── train_vacancy_model.py         # Trains models for vacancy property prediction
└── requirements.txt        # Dependencies for the project
```

## Description of Key Files
### Mechanical Property Prediction (`mechanical_prediction`)

- **get_mechanical_features.py**: Extracts features related to mechanical properties from given structure files. This includes calculating atomic environments and descriptors required for model training and prediction.
  
- **predict_mechanical_properties.py**: Loads trained models and makes predictions of mechanical properties (e.g., Elastic Modulus, Shear Modulus, Bulk Modulus) for each carbon atom in the provided structure file.
  
- **train_mechanical_model.py**: Trains machine learning models for predicting mechanical properties. It loads data, preprocesses features, and saves trained models in the `models/mechanical_prediction` directory for later use.

### Vacancy Formation Energy Prediction (`vacancy_prediction`)

- **get_vacancy_features.py**: Extracts features required for predicting vacancy formation energy in metal carbides. This includes descriptors related to local atomic environments and structural characteristics around potential vacancy sites.
  
- **predict_vacancy_properties.py**: Loads pre-trained models to predict vacancy formation energies at each carbon site in a given structure file.
  
- **train_vacancy_model.py**: Trains machine learning models for vacancy formation energy prediction. It handles data loading, feature extraction, model training, and saves the resulting models to the `models/vacancy_prediction` directory.

## Models

The trained models are stored in the `models` directory:
- `models/mechanical_prediction`: Stores models for mechanical property predictions.
- `models/vacancy_prediction`: Stores models for vacancy formation energy predictions.

## Usage

1. **Feature Extraction**: Use the feature extraction scripts (`get_mechanical_features.py` and `get_vacancy_features.py`) to obtain relevant descriptors from your structure files.
  
2. **Prediction**: Run the prediction scripts (`predict_mechanical_properties.py` and `predict_vacancy_properties.py`) on extracted features to obtain predictions.

3. **Training**: To retrain models with new data, use the training scripts (`train_mechanical_model.py` and `train_vacancy_model.py`).

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt


