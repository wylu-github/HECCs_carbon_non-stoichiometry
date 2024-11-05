# HECCs_carbon_vacancies
# Material Property Prediction

This repository contains tools for predicting the mechanical properties and carbon vacancy formation energies of metal carbide ceramics. The repository includes machine learning models and feature extraction functions organized to enable efficient predictions based on input crystal structures.

├── data │ └── structures # Directory containing input structure files for predictions │ ├── 2-metal-sqs1 # 2-metal SQS structure files │ ├── 2-metal-sqs2 │ ├── 3-metal-sqs1 # 3-metal SQS structure files │ ├── 3-metal-sqs2 │ ├── 3-metal-sqs3 │ └── 4-metal-sqs ├── mechanical_prediction # Directory containing scripts for mechanical property prediction │ └── pycache ├── models # Directory containing trained models for predictions │ ├── mechanical_prediction # Trained models for predicting mechanical properties │ └── vacancy_prediction # Trained models for predicting vacancy formation energies └── vacancy_prediction # Directory containing scripts for vacancy prediction └── pycache

markdown
复制代码

### Folders

- **data/structures**: Contains the structure files for various metal compositions, organized by metal count and sequence.
- **mechanical_prediction**: Contains Python scripts for predicting mechanical properties (Elastic Modulus, Shear Modulus, Bulk Modulus) of input structures.
- **models**: Contains pre-trained machine learning models. The models are stored separately for mechanical property prediction and vacancy prediction.
- **vacancy_prediction**: Contains Python scripts for predicting carbon vacancy formation energy for given structures.

## Installation

Clone the repository and ensure you have the required Python libraries installed. You can install dependencies with:

```bash
pip install -r requirements.txt
Requirements:

Python 3.8+
Scikit-Learn
Pymatgen
Pandas
Numpy
Usage
Predicting Mechanical Properties
To predict mechanical properties for a specific structure, place the .vasp file in the appropriate folder within data/structures, then run the mechanical prediction script as follows:

bash
复制代码
python mechanical_prediction/predict_mechanical.py --structure data/structures/3-metal-sqs1/POSCAR.vasp
The script will output the predicted Elastic Modulus (E), Shear Modulus (G), and Bulk Modulus (B) for each carbon atom site in the given structure.

Predicting Vacancy Formation Energies
To predict carbon vacancy formation energies, use the vacancy prediction script as follows:

bash
复制代码
python vacancy_prediction/predict_vacancy.py --structure data/structures/3-metal-sqs1/POSCAR.vasp
The script will output the predicted vacancy formation energy for each carbon atom site.

Models
Trained models are stored in the models folder. Each model is saved in .pkl format and can be loaded directly for predictions:

mechanical_prediction: Contains models for mechanical properties.
vacancy_prediction: Contains models for predicting vacancy formation energy.
Contributing
Feel free to open issues or submit pull requests if you would like to contribute to the project.

License
This project is licensed under the MIT License.

复制代码






