import os
import pandas as pd
import numpy as np
import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"


def load_to_csv(working_set, new_file_name):
    path = f"Dataset_Files/{new_file_name}.csv"
    working_set.to_csv(path, index=False)


def load_from_csv(csv_file):
    path = f"Dataset_Files/{csv_file}.csv"
    return pd.read_csv(path)


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


def populate_dataset(csv_file, new_file_name):
    working_set = load_from_csv(csv_file)

    for index, row in working_set.iterrows():
        protein_name = get_pubchem_protein_name_using_accession(row["Protein_Accession"])
        if protein_name is not None:
            working_set.at[index, 'Protein_Name'] = protein_name
            print(f"Processed: {index}")
        else:
            working_set.at[index, 'Protein_Name'] = ''
            print(f"Skipped: {index}")

        # if index == 10:
        #      break

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


if __name__ == "__main__":
    # working_set = get_protein_accessions("AlphaFold_Proteins/")
    # load_to_csv(working_set, "Dataset_Populated")
    populate_dataset("Dataset_Populated","Dataset_Populated_Name")

