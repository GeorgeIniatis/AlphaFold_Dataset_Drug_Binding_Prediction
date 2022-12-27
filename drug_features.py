from utils import *
import requests
import base64
import math

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
                    descriptors_dictionary[descriptor] = np.NaN

        return descriptors_dictionary
    else:
        return None


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

        if (index != 0) and (index % 10000 == 0):
            load_to_csv(working_set, f"{new_file_name}_{index}")

    print("Loading everything to csv file")
    load_to_csv(working_set, f"{new_file_name}")


def fingerprint_to_binary(fingerprint):
    decoded = base64.b64decode(fingerprint)

    if len(decoded * 8) == 920:
        return "".join(["{:08b}".format(x) for x in decoded])
    else:
        return None


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


# Takes a random sample of 100 drugs and checks that their descriptors match the ones in PubChem
def sanity_check(csv_file, sample=100):
    working_set = load_from_csv(csv_file)
    sample = working_set.sample(sample)

    for index, row in sample.iterrows():
        drug_cid = row["Drug_CID"]
        drug_descriptors = get_chemical_descriptors(drug_cid)
        for descriptor in DESCRIPTORS:
            if descriptor != "Fingerprint2D":
                if not np.isnan(row[descriptor]):
                    if not math.isclose(row[descriptor], float(drug_descriptors[descriptor])):
                        print(row[descriptor])
                        print(drug_descriptors[descriptor])
                        return "Fail Descriptors"
            else:
                if row[descriptor] != drug_descriptors[descriptor]:
                    return "Fail Fingerprint"
    return "Pass"


if __name__ == "__main__":
    populate_drug_descriptors("Unique_Drugs_List", "Unique_Drugs_Populated")
    populate_one_hot_encoding_fingerprint("Unique_Drugs_Populated", "Unique_Drugs_Populated_Fingerprints_Expanded")
    print(f"Sanity Check: {sanity_check('Unique_Drugs_Populated')}")
