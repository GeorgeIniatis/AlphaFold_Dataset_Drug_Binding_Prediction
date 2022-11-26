from utils import *
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import os
import h5py
import json

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"


def get_pubchem_protein_name_using_accession(accession):
    response = requests.get(BASE + f"protein/accession/{accession}/summary/json")
    if response.status_code == 200:
        return response.json()["ProteinSummaries"]["ProteinSummary"][0]["Name"]
    else:
        return None


# Expects a directory of AlphaFold protein pdb files
def populate_protein_accessions_pubchem_names_sequences(path_to_pdb_files, new_file_name):
    data = []
    columns = ["Protein_Accession", "Protein_Name", "Protein_Sequence"]

    index = 0
    for file_name in os.listdir(path_to_pdb_files):
        accession = file_name.split("-")[1]
        protein_name = get_pubchem_protein_name_using_accession(accession)
        with open(f"{path_to_pdb_files}{file_name}") as file:
            for record in SeqIO.parse(file, 'pdb-seqres'):
                sequence = record.seq
                data.append([accession, protein_name, sequence])

        print(f"Processed: {index}")
        index += 1

    print("Loading everything to csv file")
    load_to_csv(pd.DataFrame(data=data, columns=columns), f"{new_file_name}")


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
    # Step 1: Run Sequence_Descriptors.R

    # Step 2: UniProt Sequence Embeddings to individual column entries
    populate_uniprot_embedding_lists_to_individual_entries("Unique_Proteins_Populated",
                                                           "Unique_Proteins_Populated_Embedding_Entries")

    # Step 3: Tripeptide Descriptors Dimensionality Reduction
    tripeptide_composition_descriptors_dimensionality_reduction("Unique_Proteins_Populated_Embedding_Entries",
                                                                "Unique_Proteins_Populated_Embedding_Entries_PCA_Reduction")
