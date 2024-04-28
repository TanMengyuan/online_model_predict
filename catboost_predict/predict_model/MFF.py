from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Load the input CSV file containing SMILES strings
input_file = "list-A.csv"  # Replace with your file path
df = pd.read_csv(input_file)

# Prepare a dictionary for each molecule's fingerprint
molecule_fp_dict = {}

# Process each SMILES string
for index, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    fp = AllChem.GetMorganFingerprint(mol, radius=2)
    molecule_fp_dict[row['smiles']] = fp.GetNonzeroElements()

# Find the union of all bits present in all fingerprints
all_bits = set()
for fp in molecule_fp_dict.values():
    all_bits.update(fp.keys())

# Create a DataFrame with zeros
mff_df = pd.DataFrame(0, index=df['smiles'], columns=all_bits)

# Fill the DataFrame with frequencies
for smiles, fp in molecule_fp_dict.items():
    for bit, freq in fp.items():
        mff_df.at[smiles, bit] = freq

# Save the DataFrame to a CSV file
output_file = "mff_per_moleculeA3.csv"  # Replace with your desired output file path
mff_df.to_csv(output_file)
