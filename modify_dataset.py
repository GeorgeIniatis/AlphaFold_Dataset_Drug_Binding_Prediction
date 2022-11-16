from urllib.request import urlopen
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def load_to_pickle(working_set, new_file_name):
    path = f"Dataset_Files/{new_file_name}.pkl"
    working_set.to_pickle(path)


def load_from_pickle(file_name):
    path = f"Dataset_Files/{file_name}.pkl"
    return pd.read_pickle(path)


def load_to_csv(working_set, new_file_name):
    path = f"Dataset_Files/{new_file_name}.csv"
    working_set.to_csv(path, index=False)


def load_from_csv(csv_file):
    path = f"Dataset_Files/{csv_file}.csv"
    return pd.read_csv(path)


def replace_with_nan(working_set, string_to_replace):
    working_set.replace(string_to_replace, np.NaN, inplace=True)


def fingerprint_to_binary(fingerprint):
    decoded = base64.b64decode(fingerprint)

    if len(decoded * 8) == 920:
        return "".join(["{:08b}".format(x) for x in decoded])
    else:
        return None


# Expects a directory of txt files created by the Amino_Acid_Descriptors.R file
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


# Expects a directory of txt files created by the Amino_Acid_Descriptors.R file
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


def sanity_check_dimensions(accession, print_information=False):
    contact_map = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Contact_Map_Files/{accession}.npy")
    descriptors = np.load(
        f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM/{accession}_Descriptors.npy")
    pssm = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM/{accession}_PSSM.npy")
    embedding = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Embeddings/{accession}.npy")

    if print_information:
        print(f"Contact Map Shape: {contact_map.shape}")
        print(f"Amino Acid Descriptors Shape: {descriptors.shape}")
        print(f"PSSM Shape: {pssm.shape}")
        print(f"UniProt Embedding Shape: {embedding.shape}")

    if contact_map.shape[0] == descriptors.shape[0] == pssm.shape[0] == embedding.shape[0]:
        return True
    else:
        return False


# Expects a csv file that contains protein accessions and sequences
def get_sequences_as_FASTA_files(csv_file, path_to_save):
    working_set = load_from_csv(csv_file)

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        protein_sequence = row["Protein_Sequence"]

        with open(f"{path_to_save}{protein_accession}.fasta", 'w') as file:
            file.write(f">{protein_accession}\n{protein_sequence}\n")

        print(f"Processed: {index}")


# Expects a directory of AlphaFold protein pdb files
def get_contact_maps_as_numpy_files(pdf_files_path, path_to_save, threshold=10.0):
    index = 0
    for file_name in os.listdir(pdf_files_path):
        accession = file_name.split("-")[1]

        structure = contactmaps.get_structure(f"{pdf_files_path}{file_name}")
        model = structure[0]
        matrix = contactmaps.ContactMap(model, threshold=threshold).matrix

        np.save(f"{path_to_save}/{accession}", matrix)

        print(f"Processed: {index}")
        index += 1


def get_uniprot_molecular_function_keywords(protein_accession):
    response = requests.get(
        f"https://rest.uniprot.org/uniprotkb/search?query=accession:{protein_accession}&fields=keyword")

    molecular_functions_list = []
    if response.status_code == 200:
        try:
            keyword_dictionaries = response.json()["results"][0]["keywords"]

            for dictionary in keyword_dictionaries:
                if dictionary["category"] == "Molecular function":
                    molecular_functions_list.append(dictionary["name"])

            if molecular_functions_list:
                return molecular_functions_list
            else:
                return None
        except:
            return None
    else:
        return None


# Expects a directory of h5y files downloaded from UniProt holding the embeddings
def get_uniprot_residue_embeddings_as_numpy_files(path, path_to_save):
    index = 0
    with h5py.File(path, "r") as file:
        for protein_accession, embedding in file.items():
            np.save(f"{path_to_save}/{protein_accession}", np.array(embedding))

            print(f"Processed: {index}")
            index += 1


# Expects a directory of h5y files downloaded from UniProt holding the embeddings
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


# Expects a directory of AlphaFold protein pdb files
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

    for index, row in working_set.iterrows():
        drug_cid = row["Drug_CID"]
        drug_descriptors = get_chemical_descriptors(drug_cid)

        if drug_descriptors is not None:
            for descriptor in DESCRIPTORS:
                working_set.at[index, descriptor] = drug_descriptors[descriptor]
            print(f"Processed: {index}")
        else:
            print(f"Skipped: {index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


def populate_one_hot_encoding_fingerprint(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)
    data = []
    empty_row = [np.NaN for i in range(881)]
    columns = [f"Fingerprint_Bit_{i + 1}" for i in range(881)]

    for index, row in working_set.iterrows():
        fingerprint = row["Fingerprint2D"]
        if pd.notna(fingerprint):
            fingerprint_binary = fingerprint_to_binary(fingerprint)
            fingerprint_list = [int(i) for i in str(fingerprint_binary)]

            # The first 32 bits are prefix,containing the bit length of the fingerprint (881 bits)
            # The last 7 bits are padding
            fingerprint_list_prefix_and_padding_removed = fingerprint_list[
                                                          32:len(fingerprint_list) - 7]

            data.append(fingerprint_list_prefix_and_padding_removed)
        else:
            data.append(empty_row)

        print(f"Processed: {index}")

    temp_dataframe = pd.DataFrame(data=data, columns=columns)
    joined_set = working_set.join(temp_dataframe)

    print("Loading everything to csv file")
    load_to_csv(joined_set, new_file_name)


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


def populate_uniprot_embedding_lists_to_individual_entries(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)
    data = []
    empty_row = [np.NaN for i in range(1024)]
    columns = [f"UniProt_Embedding_{i + 1}" for i in range(1024)]

    for index, row in working_set.iterrows():
        uniprot_embeddings = row["UniProt_Sequence_Embedding"]
        if pd.notna(uniprot_embeddings):
            uniprot_embeddings_list = json.loads(uniprot_embeddings)
            data.append(uniprot_embeddings_list)
        else:
            data.append(empty_row)

        print(f"Processed: {index}")

    temp_dataframe = pd.DataFrame(data=data, columns=columns)
    joined_set = working_set.join(temp_dataframe)

    print("Loading everything to csv file")
    load_to_csv(joined_set, new_file_name)


def populate_uniprot_molecular_function_keywords(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        molecular_functions_list = get_uniprot_molecular_function_keywords(protein_accession)

        if molecular_functions_list is not None:
            working_set.at[index, "UniProt_Molecular_Functions"] = json.dumps(molecular_functions_list)
            print(f"Processed: {index}")
        else:
            print(f"Skipped: {index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


def populate_one_hot_encoding_molecular_function_keywords(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)
    working_set['UniProt_Molecular_Functions_JSON'] = ''

    for index, row in working_set.iterrows():
        molecular_functions_string = row['UniProt_Molecular_Functions']
        if pd.notna(molecular_functions_string):
            working_set.at[index, 'UniProt_Molecular_Functions_JSON'] = json.loads(molecular_functions_string)

    one_hot_side_effects = working_set['UniProt_Molecular_Functions_JSON'].str.join('|').str.get_dummies().add_prefix(
        'Molecular_Function_')
    joined_set = working_set.join(one_hot_side_effects)
    joined_set.drop(columns=['UniProt_Molecular_Functions_JSON'], inplace=True)

    print("Loading everything to csv file")
    load_to_csv(joined_set, f"{new_file_name}")


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


# Tripeptide Composition Descriptors were incredibly sparse, so it was decided to reduce their dimensionality with PCA
def tripeptide_composition_descriptors_dimensionality_reduction(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)

    scaler = StandardScaler()
    scaler.fit(working_set.loc[:, "AAA":"VVV"])
    scaled_data = scaler.transform(working_set.loc[:, "AAA":"VVV"])

    # Choose a number of components that captures 95% of the variance
    pca = PCA(n_components=0.95)
    pca.fit(scaled_data)

    data = pca.transform(scaled_data)
    columns = [f"Tripeptide_Composition_PCA_Component_{i + 1}" for i in range(data.shape[1])]
    temp_dataframe = pd.DataFrame(data=data, columns=columns)

    working_set = working_set.join(temp_dataframe)

    print("Loading everything to csv file")
    load_to_csv(working_set, new_file_name)


if __name__ == "__main__":
    # get_contact_maps_as_numpy_files("AlphaFold_Proteins/", "Dataset_Files/Contact_Map_Files/", 10.0)
    # get_sequences_as_FASTA_files("Backups/AlphaFold_Proteins_Accessions_Names_UniProt_Embeddings_Sequences",
    #                              "Dataset_Files/Sequence_FASTA_Files/")
    # amino_acid_descriptors_to_numpy_files("Dataset_Files/Amino_Acid_Descriptors_Text_Files/",
    #                                       "Dataset_Files/Amino_Acid_Descriptors_And_PSSM/")
    # amino_acid_pssms_to_numpy_files("Dataset_Files/Amino_Acids_PSSM_Text_Files/",
    #                                 "Dataset_Files/Amino_Acid_Descriptors_And_PSSM/")
    # print(sanity_check_dimensions("Q30201"))
    # populate_uniprot_molecular_function_keywords("Unique_Proteins_UniProt_Embeddings_Sequences_Descriptors",
    #                                              "Unique_Proteins_UniProt_Embeddings_Sequences_Descriptors_Molecular_Functions")
    # get_uniprot_residue_embeddings_as_numpy_files("D:/AlphaFold_Project_Backups/per-residue.h5",
    #                                               "Dataset_Files/Amino_Acid_Embeddings/")
    # populate_drug_descriptors("Unique_Drugs_Populated", "Unique_Drugs_Populated_Fixed")
    # populate_one_hot_encoding_fingerprint("Unique_Drugs_Populated", "Unique_Drugs_Populated_One_Hot")
    # populate_one_hot_encoding_molecular_function_keywords("Unique_Proteins_Populated",
    #                                                       "Unique_Proteins_Populated_One_Hot")
    # populate_uniprot_embedding_lists_to_individual_entries("Unique_Proteins_Populated",
    #                                                        "Unique_Proteins_Populated_Embedding_Entries")
    tripeptide_composition_descriptors_dimensionality_reduction("Unique_Proteins_Populated",
                                                                "Unique_Proteins_Populated_PCA")
