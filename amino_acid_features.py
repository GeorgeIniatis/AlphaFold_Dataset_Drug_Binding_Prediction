from utils import *
import os
import contactmaps
import h5py
import numpy as np


# Expects a csv file that contains protein accessions and sequences
def get_sequences_as_FASTA_files(csv_file, path_to_save):
    working_set = load_from_csv(csv_file)

    for index, row in working_set.iterrows():
        protein_accession = row["Protein_Accession"]
        protein_sequence = row["Protein_Sequence"]

        with open(f"{path_to_save}{protein_accession}.fasta", 'w') as file:
            file.write(f">{protein_accession}\n{protein_sequence}\n")

        print(f"Processed: {index}")


# Expects a directory of txt files created by the Amino_Acid_Descriptors.R file
def amino_acid_descriptors_to_numpy_files(path, path_to_save):
    index = 0
    for file_name in os.listdir(path):
        protein_accession = file_name.split(".")[0]
        matrix = []
        with open(f"{path}/{file_name}") as file:
            array = file.read().split(",")
            array[-1] = array[-1].split("\n")[0]

            for i in range(int(len(array) / 66)):
                matrix.append(array[i * 66:(i + 1) * 66])

            numpy_array = np.array(matrix, dtype=float)
            load_to_numpy(numpy_array, f"{path_to_save}/{protein_accession}_Descriptors")

            print(f"Processed: {index}")
            index += 1


# Expects a directory of txt files created by the Amino_Acid_Descriptors.R file
def amino_acid_pssms_to_numpy_files(path, path_to_save):
    index = 0
    for file_name in os.listdir(path):
        protein_accession = file_name.split(".")[0]
        matrix = []
        with open(f"{path}/{file_name}") as file:
            array = file.read().split()

            for i in range(int(len(array) / 20)):
                matrix.append(array[i * 20:(i + 1) * 20])

            numpy_array = np.array(matrix, dtype=float)
            load_to_numpy(numpy_array, f"{path_to_save}/{protein_accession}_PSSM")

            print(f"Processed: {index}")
            index += 1


# Expects a directory of h5y files downloaded from UniProt holding the embeddings
def get_uniprot_residue_embeddings_as_numpy_files(path, path_to_save):
    index = 0
    with h5py.File(path, "r") as file:
        for protein_accession, embedding in file.items():
            load_to_numpy(np.array(embedding), f"{path_to_save}/{protein_accession}")

            print(f"Processed: {index}")
            index += 1


# Expects a directory of AlphaFold protein pdb files
def get_contact_maps_as_numpy_files(pdf_files_path, path_to_save, threshold=10.0):
    index = 0
    for file_name in os.listdir(pdf_files_path):
        if "F1" in file_name:
            accession = file_name.split("-")[1]

            structure = contactmaps.get_structure(f"{pdf_files_path}/{file_name}")
            model = structure[0]
            matrix = contactmaps.ContactMap(model, threshold=threshold).matrix

            load_to_numpy(matrix, f"{path_to_save}/{accession}")

            print(f"Processed: {index}")
            index += 1


def sanity_check_dimensions(accession, print_information=False):
    try:
        contact_map = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Contact_Map_Files/{accession}.npy")
        descriptors = np.load(
            f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM/{accession}_Descriptors.npy")
        pssm = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM/{accession}_PSSM.npy")
        embedding = np.load(f"Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Embeddings/{accession}.npy")

        if print_information:
            print(f"Contact Map Shape: {contact_map.shape}")
            print(f"Amino Acid Descriptors Shape: {descriptors.shape}")
            print(f"PSSM Shape: {pssm.shape}")
            print(f"UniProt Per-Residue Embedding Shape: {embedding.shape}")

        if contact_map.shape[0] == descriptors.shape[0] == pssm.shape[0] == embedding.shape[0]:
            return True
        else:
            return False
    except FileNotFoundError:
        return "FileNotFound"


if __name__ == "__main__":
    # Step 1: Get FASTA Files
    get_sequences_as_FASTA_files("AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped",
                                 "Dataset_Files/Protein_Graph_Data/raw/Sequence_FASTA_Files/")

    # Step 2: Run Amino_Acid_Descriptors.R

    # Step 3: Convert Amino Acid Descriptors txt files to numpy files
    print("Amino Acid Descriptors To Numpy")
    amino_acid_descriptors_to_numpy_files("Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_Text_Files",
                                          "Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM")

    # Step 4: Convert PSSMs txt files to numpy files
    print("PSSMs To Numpy")
    amino_acid_pssms_to_numpy_files("Dataset_Files/Protein_Graph_Data/raw/Amino_Acids_PSSM_Text_Files",
                                    "Protein_Graph_Data/raw/Amino_Acid_Descriptors_And_PSSM")

    # Step 5: Get UniProt Residue Embeddings
    print("UniProt Residue Embeddings To Numpy")
    get_uniprot_residue_embeddings_as_numpy_files("Dataset_Files/Protein_Graph_Data/raw/per-residue.h5",
                                                  "Protein_Graph_Data/raw/Amino_Acid_Embeddings")

    # Step 6: Get Contact Maps
    print("Contact Maps To Numpy")
    get_contact_maps_as_numpy_files("Dataset_Files/AlphaFold_Proteins",
                                    "Protein_Graph_Data/raw/Contact_Map_Files")
