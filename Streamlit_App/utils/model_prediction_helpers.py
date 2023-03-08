from joblib import load
import gdown
import base64
import requests
import pandas as pd
import streamlit as st

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"

DESCRIPTORS = ['MolecularWeight', 'XLogP', 'ExactMass', 'MonoisotopicMass', 'TPSA', 'Complexity', 'HBondDonorCount',
               'HBondAcceptorCount', 'RotatableBondCount', 'HeavyAtomCount', 'AtomStereoCount',
               'DefinedAtomStereoCount', 'Volume3D', 'XStericQuadrupole3D', 'YStericQuadrupole3D',
               'ZStericQuadrupole3D', 'FeatureCount3D', 'FeatureAcceptorCount3D', 'FeatureDonorCount3D',
               'FeatureCationCount3D', 'FeatureRingCount3D', 'FeatureHydrophobeCount3D', 'ConformerModelRMSD3D',
               'EffectiveRotorCount3D', 'ConformerCount3D', 'Fingerprint2D']
DESCRIPTORS_STRING = ','.join(DESCRIPTORS)

CLASSIFICATION_FINGERPRINT_BITS = ['Fingerprint_Bit_3', 'Fingerprint_Bit_17', 'Fingerprint_Bit_24',
                                   'Fingerprint_Bit_144', 'Fingerprint_Bit_146', 'Fingerprint_Bit_147',
                                   'Fingerprint_Bit_181', 'Fingerprint_Bit_182', 'Fingerprint_Bit_187',
                                   'Fingerprint_Bit_193', 'Fingerprint_Bit_200', 'Fingerprint_Bit_242',
                                   'Fingerprint_Bit_249', 'Fingerprint_Bit_288', 'Fingerprint_Bit_294',
                                   'Fingerprint_Bit_309', 'Fingerprint_Bit_342', 'Fingerprint_Bit_346',
                                   'Fingerprint_Bit_354', 'Fingerprint_Bit_358', 'Fingerprint_Bit_367',
                                   'Fingerprint_Bit_375', 'Fingerprint_Bit_378', 'Fingerprint_Bit_380',
                                   'Fingerprint_Bit_382', 'Fingerprint_Bit_383', 'Fingerprint_Bit_392',
                                   'Fingerprint_Bit_393', 'Fingerprint_Bit_406', 'Fingerprint_Bit_432',
                                   'Fingerprint_Bit_436', 'Fingerprint_Bit_448', 'Fingerprint_Bit_452',
                                   'Fingerprint_Bit_483', 'Fingerprint_Bit_496', 'Fingerprint_Bit_503',
                                   'Fingerprint_Bit_507', 'Fingerprint_Bit_529', 'Fingerprint_Bit_540',
                                   'Fingerprint_Bit_543', 'Fingerprint_Bit_549', 'Fingerprint_Bit_554',
                                   'Fingerprint_Bit_570', 'Fingerprint_Bit_577', 'Fingerprint_Bit_578',
                                   'Fingerprint_Bit_580', 'Fingerprint_Bit_581', 'Fingerprint_Bit_594',
                                   'Fingerprint_Bit_598', 'Fingerprint_Bit_612', 'Fingerprint_Bit_615',
                                   'Fingerprint_Bit_624', 'Fingerprint_Bit_644', 'Fingerprint_Bit_646',
                                   'Fingerprint_Bit_647', 'Fingerprint_Bit_652', 'Fingerprint_Bit_672',
                                   'Fingerprint_Bit_673', 'Fingerprint_Bit_685', 'Fingerprint_Bit_693',
                                   'Fingerprint_Bit_697', 'Fingerprint_Bit_698', 'Fingerprint_Bit_699',
                                   'Fingerprint_Bit_700', 'Fingerprint_Bit_705', 'Fingerprint_Bit_713']

REGRESSION_FINGERPRINT_BITS = ['Fingerprint_Bit_38', 'Fingerprint_Bit_39', 'Fingerprint_Bit_151',
                               'Fingerprint_Bit_154', 'Fingerprint_Bit_158', 'Fingerprint_Bit_160',
                               'Fingerprint_Bit_168', 'Fingerprint_Bit_182', 'Fingerprint_Bit_186',
                               'Fingerprint_Bit_189', 'Fingerprint_Bit_193', 'Fingerprint_Bit_200',
                               'Fingerprint_Bit_207', 'Fingerprint_Bit_244', 'Fingerprint_Bit_249',
                               'Fingerprint_Bit_252', 'Fingerprint_Bit_259', 'Fingerprint_Bit_294',
                               'Fingerprint_Bit_335', 'Fingerprint_Bit_336', 'Fingerprint_Bit_337',
                               'Fingerprint_Bit_339', 'Fingerprint_Bit_342', 'Fingerprint_Bit_347',
                               'Fingerprint_Bit_365', 'Fingerprint_Bit_378', 'Fingerprint_Bit_381',
                               'Fingerprint_Bit_387', 'Fingerprint_Bit_389', 'Fingerprint_Bit_393',
                               'Fingerprint_Bit_420', 'Fingerprint_Bit_421', 'Fingerprint_Bit_430',
                               'Fingerprint_Bit_432', 'Fingerprint_Bit_433', 'Fingerprint_Bit_440',
                               'Fingerprint_Bit_446', 'Fingerprint_Bit_450', 'Fingerprint_Bit_478',
                               'Fingerprint_Bit_480', 'Fingerprint_Bit_484', 'Fingerprint_Bit_486',
                               'Fingerprint_Bit_489', 'Fingerprint_Bit_494', 'Fingerprint_Bit_501',
                               'Fingerprint_Bit_517', 'Fingerprint_Bit_520', 'Fingerprint_Bit_525',
                               'Fingerprint_Bit_529', 'Fingerprint_Bit_534', 'Fingerprint_Bit_536',
                               'Fingerprint_Bit_547', 'Fingerprint_Bit_554', 'Fingerprint_Bit_578',
                               'Fingerprint_Bit_594', 'Fingerprint_Bit_598', 'Fingerprint_Bit_600',
                               'Fingerprint_Bit_613', 'Fingerprint_Bit_615', 'Fingerprint_Bit_616',
                               'Fingerprint_Bit_618', 'Fingerprint_Bit_630', 'Fingerprint_Bit_631',
                               'Fingerprint_Bit_633', 'Fingerprint_Bit_640', 'Fingerprint_Bit_646',
                               'Fingerprint_Bit_647', 'Fingerprint_Bit_659', 'Fingerprint_Bit_673',
                               'Fingerprint_Bit_683', 'Fingerprint_Bit_686', 'Fingerprint_Bit_687',
                               'Fingerprint_Bit_694', 'Fingerprint_Bit_696', 'Fingerprint_Bit_697',
                               'Fingerprint_Bit_698', 'Fingerprint_Bit_699', 'Fingerprint_Bit_700',
                               'Fingerprint_Bit_703', 'Fingerprint_Bit_704', 'Fingerprint_Bit_705',
                               'Fingerprint_Bit_713', 'Fingerprint_Bit_729', 'Fingerprint_Bit_738',
                               'Fingerprint_Bit_739', 'Fingerprint_Bit_780', 'Fingerprint_Bit_785',
                               'Fingerprint_Bit_795', 'Fingerprint_Bit_816', 'Fingerprint_Bit_834']

classification_model_name_to_file = {
    '-': '-',
    'Dummy Classifier': 'dc.joblib',
    'Logistic Regression': 'optimised_lr.joblib',
    'Linear Support Vector Classification': 'optimised_lsvc.joblib',
    'K-Nearest Neighbour Classifier': 'optimised_knnc.joblib',
    'Decision Tree Classifier': 'optimised_dtc.joblib',
    'Random Forest Classifier': 'optimised_rfc.joblib',
    'Stochastic Gradient Descent Classifier': 'optimised_sgdc.joblib',
}

regression_model_name_to_file = {
    '-': '-',
    'Dummy Regressor': 'dr.joblib',
    'Linear Regression': 'lr.joblib',
    'Linear Support Vector Regression': 'optimised_lsvr.joblib',
    'K-Nearest Neighbour Regressor': 'optimised_knnr.joblib',
    'Decision Tree Regressor': 'optimised_dtr.joblib',
    'Random Forest Regressor': 'optimised_rfr.joblib',
    'Stochastic Gradient Descent Regressor': 'optimised_sgdr.joblib',
}


@st.cache_data
def load_model(chosen_model, category, key):
    if category == "Classification":
        if chosen_model not in ["K-Nearest Neighbour Classifier", "Random Forest Classifier"]:
            model = load(f"Streamlit_App/data/{key}/{category}/{classification_model_name_to_file[chosen_model]}")
        else:
            model = load_model_from_drive(chosen_model, category, key)
    elif category == "Regression":
        if chosen_model not in ["K-Nearest Neighbour Regressor", "Random Forest Regressor"]:
            model = load(f"Streamlit_App/data/{key}/{category}/{regression_model_name_to_file[chosen_model]}")
        else:
            model = load_model_from_drive(chosen_model, category, key)
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")

    return model


@st.cache_data
def load_model_from_drive(chosen_model, category, key):
    if key == "Baseline_Models":
        if chosen_model == "K-Nearest Neighbour Classifier":
            url = "https://drive.google.com/uc?id=1OhdNT-QB3vxOT-yp_q9aGMCT0iEZ3moV"

        elif chosen_model == "Random Forest Classifier":
            url = "https://drive.google.com/uc?id=1BqjASdzHNMwqd1yC0mkRsWqmMxL6srB7"

        elif chosen_model == "K-Nearest Neighbour Regressor":
            url = "https://drive.google.com/uc?id=1yYDRQQrQu6rtRQ5u8iNskPKPEIQRZQnX"

        elif chosen_model == "Random Forest Regressor":
            url = "https://drive.google.com/uc?id=1quHw8SUjFyKbrsXXW2aenV_zKj5sgU8M"

        else:
            raise ValueError("Invalid Model.")
    elif key == "Enhanced_Models":
        if chosen_model == "K-Nearest Neighbour Classifier":
            url = "https://drive.google.com/uc?id=1LAa8BfmUUTcsFVgdpPZXJYA-PX-cXg3f"

        elif chosen_model == "Random Forest Classifier":
            url = "https://drive.google.com/uc?id=1CGABR4W9b72pckwNMJt9ntq7VGypp7K8"

        elif chosen_model == "K-Nearest Neighbour Regressor":
            url = "https://drive.google.com/uc?id=1Kpv9vz5Pis-FQA5jjfmE8FweTnEyP6oy"

        elif chosen_model == "Random Forest Regressor":
            url = "https://drive.google.com/uc?id=1Y7wMX30WKrwEh_l0byAGf5HxiGM8jZDh"

        else:
            raise ValueError("Invalid Model.")
    else:
        raise ValueError("Invalid key. Please choose 'Baseline_Models' or 'Enhanced_Models'")

    if category == "Classification":
        output = f"Streamlit_App/data/{key}/{category}/{classification_model_name_to_file[chosen_model]}"
    elif category == "Regression":
        output = f"Streamlit_App/data/{key}/{category}/{regression_model_name_to_file[chosen_model]}"
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")

    try:
        model = load(output)
    except FileNotFoundError:
        gdown.download(url, output, quiet=True)
        model = load(output)

    return model


def get_protein_descriptors(protein_accession, category, key):
    if category == "Classification":
        proteins = pd.read_pickle("Streamlit_App/data/Datasets/Classification_Proteins.pkl")
        descriptors = proteins[proteins["Protein_Accession"] == protein_accession].loc[:, "LK":].squeeze()
    elif category == "Regression":
        proteins = pd.read_pickle("Streamlit_App/data/Datasets/Regression_Proteins.pkl")
        descriptors = proteins[proteins["Protein_Accession"] == protein_accession].loc[:, "D":].squeeze()
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")

    if key == "Enhanced_Models":
        embeddings = pd.read_pickle("Streamlit_App/data/Datasets/Protein_Embeddings.pkl")
        structure_embedding = embeddings[embeddings["Protein_Accession"] == protein_accession].loc[:,
                              "Structure_Embedding_0":].squeeze()

        descriptors = pd.concat(objs=[descriptors, structure_embedding])

    return descriptors


def get_chemical_descriptors(pubchem_cid, category):
    response = requests.get(
        BASE + f"compound/cid/{pubchem_cid}/property/{DESCRIPTORS_STRING}/json")
    if response.status_code == 200:
        descriptors_dictionary = response.json()['PropertyTable']['Properties'][0]
        del descriptors_dictionary['CID']

        # Some descriptors are not available for a few rare compounds
        if len(descriptors_dictionary.keys()) != len(DESCRIPTORS):
            for descriptor in DESCRIPTORS:
                if descriptor not in descriptors_dictionary.keys():
                    return f"{descriptor} not available for PubChem Compound!"

        descriptors = pd.Series(descriptors_dictionary)
        descriptors = one_hot_encoding_fingerprint(descriptors, category)

        return descriptors
    else:
        return f"PubChem Compound CID does not exist!"


def fingerprint_to_binary(fingerprint):
    decoded = base64.b64decode(fingerprint)

    if len(decoded * 8) == 920:
        return "".join(["{:08b}".format(x) for x in decoded])
    else:
        return None


def one_hot_encoding_fingerprint(descriptors, category):
    fingerprint = descriptors["Fingerprint2D"]
    fingerprint_binary = fingerprint_to_binary(fingerprint)
    fingerprint_list = [int(i) for i in str(fingerprint_binary)]

    # The first 32 bits are prefix,containing the bit length of the fingerprint (881 bits)
    # The last 7 bits are padding
    fingerprint_list_prefix_and_padding_removed = fingerprint_list[32:len(fingerprint_list) - 7]

    columns = [f"Fingerprint_Bit_{i + 1}" for i in range(881)]

    temp = pd.Series(data=fingerprint_list_prefix_and_padding_removed, index=columns)

    if category == "Classification":
        temp = temp[CLASSIFICATION_FINGERPRINT_BITS]
    elif category == "Regression":
        temp = temp[REGRESSION_FINGERPRINT_BITS]
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")

    joined_set = pd.concat(objs=[descriptors, temp])
    joined_set.drop(index=["Fingerprint2D"], inplace=True)
    return joined_set
