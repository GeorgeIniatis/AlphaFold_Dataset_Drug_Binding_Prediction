import json
import urllib
from urllib.request import urlopen
import os
import pandas as pd
import numpy as np
import requests
import ijson
import base64
import h5py

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
DESCRIPTORS = ['MolecularWeight', 'XLogP', 'ExactMass', 'MonoisotopicMass', 'TPSA', 'Complexity', 'Charge',
               'HBondDonorCount',
               'HBondAcceptorCount', 'RotatableBondCount', 'HeavyAtomCount', 'IsotopeAtomCount', 'AtomStereoCount',
               'DefinedAtomStereoCount', 'UndefinedAtomStereoCount', 'BondStereoCount', 'DefinedBondStereoCount',
               'UndefinedBondStereoCount', 'CovalentUnitCount', 'Volume3D', 'XStericQuadrupole3D',
               'YStericQuadrupole3D',
               'ZStericQuadrupole3D', 'FeatureCount3D', 'FeatureAcceptorCount3D', 'FeatureDonorCount3D',
               'FeatureAnionCount3D', 'FeatureCationCount3D', 'FeatureRingCount3D', 'FeatureHydrophobeCount3D',
               'ConformerModelRMSD3D', 'EffectiveRotorCount3D', 'ConformerCount3D', 'Fingerprint2D']
DESCRIPTORS_STRING = ','.join(DESCRIPTORS)


def load_to_csv(working_set, new_file_name):
    path = f"Dataset_Files/{new_file_name}.csv"
    working_set.to_csv(path, index=False)


def load_from_csv(csv_file):
    path = f"Dataset_Files/{csv_file}.csv"
    return pd.read_csv(path)


def fingerprint_to_binary(fingerprint):
    decoded = base64.b64decode(fingerprint)

    if len(decoded * 8) == 920:
        return "".join(["{:08b}".format(x) for x in decoded])
    else:
        return None


def get_uniprot_sequence_embeddings_as_dataframe(path):
    data = []
    columns = ["Protein_Accession", "UniProt_Sequence_Embedding"]

    with h5py.File(path, "r") as file:
        for protein_accession, embedding in file.items():
            data.append([protein_accession, np.array(embedding)])

    return pd.DataFrame(data=data, columns=columns)


def get_uniprot_sequence_embedding_for_protein(sequence_embeddings_sorted, accession):
    index_retrieved = np.searchsorted(sequence_embeddings_sorted['Protein_Accession'], accession)
    if index_retrieved < len(sequence_embeddings_sorted):
        if accession == sequence_embeddings_sorted.iloc[index_retrieved]['Protein_Accession']:
            return sequence_embeddings_sorted.iloc[index_retrieved]['UniProt_Sequence_Embedding']
    return None


def get_protein_accessions(path):
    protein_accessions = []
    for file_name in os.listdir(path):
        protein_accessions.append(file_name.split("-")[1])

    return pd.DataFrame(data=protein_accessions, columns=["Protein_Accession"])


def get_pubchem_protein_name_using_accession(accession):
    response = requests.get(BASE + f"protein/accession/{accession}/summary/json")
    if response.status_code == 200:
        return response.json()["ProteinSummaries"]["ProteinSummary"][0]["Name"]
    else:
        return None


def get_pubchem_drug_interactions(accession, max_number_of_interactions):
    drug_interactions = []
    drugs_added = []
    count = 0

    try:
        file = urlopen(BASE + f"protein/accession/{accession}/concise/json")
        rows = ijson.items(file, 'Table.Row.item')
        for row in rows:
            cell = row["Cell"]
            cid = cell[2]
            if cid not in drugs_added:
                activity_outcome = cell[3]
                if activity_outcome in ["Active", "Inactive"]:
                    activity_value = cell[5]
                    activity_name = cell[6]

                    drug_interactions.append([cid, activity_outcome, activity_value, activity_name])
                    drugs_added.append(cid)
                    count += 1
                    if count == max_number_of_interactions:
                        return drug_interactions
        return drug_interactions
    except:
        return None


def get_chemical_descriptors(pubchem_cid):
    response = requests.get(
        BASE + f"compound/cid/{pubchem_cid}/property/{DESCRIPTORS_STRING}/json")
    if response.status_code == 200:
        descriptors_dictionary = response.json()['PropertyTable']['Properties'][0]
        del descriptors_dictionary['CID']

        # Some descriptors are not available for a few rare compounds
        if len(descriptors_dictionary.keys()) != len(DESCRIPTORS):
            for descriptor in DESCRIPTORS:
                if descriptor not in descriptors_dictionary.keys():
                    descriptors_dictionary[descriptor] = '-'

        return descriptors_dictionary
    else:
        return None


def populate_dtis(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)

    data = []
    columns = ["Protein_Accession", "Protein_Name", "Drug_CID", "Activity_Outcome", "Activity_Value", "Activity_Name"]

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        protein_name = row["Protein_Name"]
        drug_interactions = get_pubchem_drug_interactions(protein_accession, 100)

        if drug_interactions is not None:
            for interaction in drug_interactions:
                drug_cid = interaction[0]
                activity_outcome = interaction[1]
                activity_value = interaction[2]
                activity_name = interaction[3]
                data.append(
                    [protein_accession, protein_name, drug_cid, activity_outcome, activity_value, activity_name])
            print(f"Processed: {index}")
        else:
            print(f"Skipped: {index}")

        if index == 10000:
            new_working_set = pd.DataFrame(data=data, columns=columns)
            load_to_csv(new_working_set, f"{new_file_name}_{index}")
            data = []

    new_working_set = pd.DataFrame(data=data, columns=columns)

    print("Loading everything to csv file")
    load_to_csv(new_working_set, f"{new_file_name}")


def populate_drug_descriptors(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)

    index_range = list(range(2000, 108000, 2000))
    for index, row in working_set.iterrows():
        if index > 86000:
            drug_cid = row["Drug_CID"]
            drug_descriptors = get_chemical_descriptors(drug_cid)

            if drug_descriptors is not None:
                for descriptor in DESCRIPTORS:
                    working_set.at[index, descriptor] = drug_descriptors[descriptor]
                print(f"Processed: {index}")
            else:
                print(f"Skipped: {index}")

            if index in index_range:
                load_to_csv(working_set, f"{new_file_name}_{index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


def populate_uniprot_sequence_embeddings(csv_file, new_file_name, path_to_embeddings):
    working_set = load_from_csv(csv_file)
    sequence_embeddings = get_uniprot_sequence_embeddings_as_dataframe(path_to_embeddings)
    sequence_embeddings_sorted = sequence_embeddings.sort_values("Protein_Accession")

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        embedding = get_uniprot_sequence_embedding_for_protein(sequence_embeddings_sorted, protein_accession)

        if embedding is not None:
            working_set.at[index, "UniProt_Sequence_Embedding"] = json.dumps(embedding.tolist())
            print(f"Processed: {index}")
        else:
            print(f"Skipped: {index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


if __name__ == "__main__":
    populate_drug_descriptors("Unique_Drugs", "Unique_Drugs_Populated")
    # populate_uniprot_sequence_embeddings("Unique_Proteins",
    #                                      "Unique_Proteins_UniProt_Embeddings",
    #                                      "Dataset_Files/uniprot_sequence_embeddings.h5")
