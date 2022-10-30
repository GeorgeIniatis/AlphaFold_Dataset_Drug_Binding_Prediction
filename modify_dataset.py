from urllib.request import urlopen
from Bio import SeqIO
import json
import os
import pandas as pd
import numpy as np
import requests
import ijson
import base64
import h5py
import contactmaps

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


def amino_acid_descriptors_to_numpy_files(path, path_to_save):
    index = 0
    for file_name in os.listdir(path):
        protein_accession = file_name.split(".")[0]
        matrix = []
        with open(f"{path}{file_name}") as file:
            array = file.read().split(",")
            array[-1] = array[-1].split("\n")[0]

            for i in range(int(len(array) / 66)):
                matrix.append(array[i * 66:(i + 1) * 66])

            numpy_array = np.array(matrix, dtype=float)
            np.save(f"{path_to_save}/{protein_accession}_Descriptors", numpy_array)

            print(f"Processed: {index}")
            index += 1


def amino_acid_pssms_to_numpy_files(path, path_to_save):
    index = 0
    for file_name in os.listdir(path):
        protein_accession = file_name.split(".")[0]
        matrix = []
        with open(f"{path}{file_name}") as file:
            array = file.read().split()

            for i in range(int(len(array) / 20)):
                matrix.append(array[i * 20:(i + 1) * 20])

            numpy_array = np.array(matrix, dtype=float)
            np.save(f"{path_to_save}/{protein_accession}_PSSM", numpy_array)

            print(f"Processed: {index}")
            index += 1


def get_sequences_as_FASTA_files(csv_file, path_to_save):
    working_set = load_from_csv(csv_file)

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        protein_sequence = row["Protein_Sequence"]

        with open(f"{path_to_save}{protein_accession}.fasta", 'w') as file:
            file.write(f">{protein_accession}\n{protein_sequence}\n")

        print(f"Processed: {index}")


def get_contact_maps_as_numpy_files(pdf_files_path, path_to_save, threshold):
    index = 0
    for file_name in os.listdir(pdf_files_path):
        accession = file_name.split("-")[1]

        structure = contactmaps.get_structure(f"{pdf_files_path}{file_name}")
        model = structure[0]
        matrix = contactmaps.ContactMap(model, threshold=threshold).matrix

        np.save(f"{path_to_save}/{accession}", matrix)

        print(f"Processed: {index}")
        index += 1


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


def get_protein_accessions_and_sequences_as_dataframe(path):
    data = []
    columns = ["Protein_Accession", "Protein_Sequence"]

    index = 0
    for file_name in os.listdir(path):
        accession = file_name.split("-")[1]
        with open(f"{path}{file_name}") as file:
            for record in SeqIO.parse(file, 'pdb-seqres'):
                sequence = record.seq
                data.append([accession, sequence])
        print(f"Processed: {index}")
        index += 1

    return pd.DataFrame(data=data, columns=columns)


def get_sequence_for_protein(protein_accessions_and_sequences_sorted, accession):
    index_retrieved = np.searchsorted(protein_accessions_and_sequences_sorted['Protein_Accession'], accession)
    if index_retrieved < len(protein_accessions_and_sequences_sorted):
        if accession == protein_accessions_and_sequences_sorted.iloc[index_retrieved]['Protein_Accession']:
            return protein_accessions_and_sequences_sorted.iloc[index_retrieved]['Protein_Sequence']
    return None


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


def populate_protein_sequences(csv_file, new_file_name, path_to_pdb_files):
    working_set = load_from_csv(csv_file)
    protein_accessions_and_sequences = get_protein_accessions_and_sequences_as_dataframe(path_to_pdb_files)
    protein_accessions_and_sequences_sorted = protein_accessions_and_sequences.sort_values("Protein_Accession")

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        sequence = get_sequence_for_protein(protein_accessions_and_sequences_sorted, protein_accession)

        if sequence is not None:
            working_set.at[index, "Protein_Sequence"] = sequence
            print(f"Processed: {index}")
        else:
            print(f"Skipped: {index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


if __name__ == "__main__":
    # get_contact_maps_as_numpy_files("AlphaFold_Proteins/", "Dataset_Files/Contact_Map_Files/", 10.0)
    # get_sequences_as_FASTA_files("Backups/AlphaFold_Proteins_Accessions_Names_UniProt_Embeddings_Sequences",
    #                              "Dataset_Files/Sequence_FASTA_Files/")
    # amino_acid_descriptors_to_numpy_files("Dataset_Files/Amino_Acid_Descriptors_Text_Files/",
    #                                       "Dataset_Files/Amino_Acid_Descriptors_And_PSSM/")
    amino_acid_pssms_to_numpy_files("Dataset_Files/Amino_Acids_PSSM_Text_Files/",
                                    "Dataset_Files/Amino_Acid_Descriptors_And_PSSM/")
