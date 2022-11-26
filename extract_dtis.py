from protein_features import populate_protein_accessions_pubchem_names_sequences, populate_uniprot_sequence_embeddings
from protein_features import populate_uniprot_molecular_function_keywords
from urllib.request import urlopen
from utils import *
import ijson

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"


def get_pubchem_drug_interactions(accession, max_number_of_interactions=100):
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

        if (index != 0) and (index % 10000 == 0):
            new_working_set = pd.DataFrame(data=data, columns=columns)
            load_to_csv(new_working_set, f"{new_file_name}_{index}")

    new_working_set = pd.DataFrame(data=data, columns=columns)
    new_working_set["Drug_CID"] = new_working_set["Drug_CID"].astype(int)

    print("Loading everything to csv file")
    load_to_csv(new_working_set, f"{new_file_name}")


if __name__ == "__main__":
    # Step 1: Get all the protein accessions, sequences and PubChem names
    populate_protein_accessions_pubchem_names_sequences("Dataset_Files/AlphaFold_Proteins/",
                                                        "AlphaFold_Proteins_Accessions_Names_Sequences")

    # Step 2: Keep only the first (F1) protein representations as these are the ones present in PubChem
    proteins = load_from_csv("AlphaFold_Proteins_Accessions_Names_Sequences")
    proteins_without_duplicates = proteins.drop_duplicates(subset=['Protein_Accession'])
    load_to_csv(proteins_without_duplicates, "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped")

    # Step 3: Extract UniProt Sequence Embeddings
    populate_uniprot_sequence_embeddings("AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped",
                                         "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings",
                                         "Dataset_Files/uniprot_sequence_embeddings.h5")

    # Step 4: Extract UniProt Molecular Functions
    populate_uniprot_molecular_function_keywords("AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings",
                                                 "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings_Molecular_Functions")

    # Step 5: Remove proteins not found in PubChem
    proteins_without_duplicates_uniprot_embeddings = load_from_csv(
        "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings_Molecular_Functions")
    proteins_without_duplicates_uniprot_embeddings_without_unknowns = proteins_without_duplicates_uniprot_embeddings.dropna(
        subset=["Protein_Name"])
    load_to_csv(proteins_without_duplicates_uniprot_embeddings_without_unknowns,
                "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings_Molecular_Functions_Unknowns_Dropped")

    # Step 6: Populate Drug-Target Interactions for each protein
    populate_dtis("AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_Unknowns_Dropped",
                  "Dataset_Populated_DTIs")

    # Step 7: Remove DTIs with Drug_CID == NaN
    dtis = load_from_csv("Dataset_Populated_DTIs")
    dtis_nan_cid_dropped = dtis.dropna(subset=['Drug_CID'])
    load_to_csv(dtis_nan_cid_dropped, "Dataset_Populated_DTIs_NaN_CID_Dropped")

    # Step 8: Separate the unique drugs and proteins
    dtis_nan_cid_dropped = load_from_csv("Dataset_Populated_DTIs_NaN_CID_Dropped")

    unique_drugs = dtis_nan_cid_dropped["Drug_CID"].unique()
    unique_drugs_dataframe = pd.DataFrame(data=unique_drugs, columns=["Drug_CID"])
    load_to_csv(unique_drugs_dataframe, "Unique_Drugs_List")

    unique_proteins = dtis_nan_cid_dropped["Protein_Accession"].unique()
    proteins_without_duplicates_without_unknowns = load_from_csv(
        "AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped_UniProt_Embeddings_Molecular_Functions_Unknowns_Dropped")
    unique_proteins_dataframe = pd.DataFrame(data=unique_proteins, columns=["Protein_Accession"])
    unique_proteins_dataframe = unique_proteins_dataframe.merge(proteins_without_duplicates_without_unknowns,
                                                                how="left")
    load_to_csv(unique_proteins_dataframe, "Unique_Proteins_List")
