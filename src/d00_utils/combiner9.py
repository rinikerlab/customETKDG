from src.d00_utils.utils import *
import time
import copy

# Just a short script to combine pdb models from different simulations.
# Probably of no further use.

t = time.time()

increment = 60

pdb_code = '1dp6'
os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'

pdb1_path = f'51_1ns_solv_0-59idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb2_path = f'51_1ns_solv_60-119idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb3_path = f'51_1ns_solv_120-179idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb4_path = f'51_1ns_solv_180-239idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb5_path = f'51_1ns_solv_240-299idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb6_path = f'51_1ns_solv_300-359idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb7_path = f'51_1ns_solv_360-419idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb8_path = f'51_1ns_solv_420-479idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb9_path = f'51_1ns_solv_480-539idx_{pdb_code}_tol1_noebounds_5400.pdb'

out_name = f'51_1ns_solv_{pdb_code}_tol1_noebounds_540.pdb'

##############################################################################

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')

smiles_mol = Chem.MolFromSmiles(smiles)

mol1 = Chem.MolFromPDBFile(pdb1_path, sanitize=True, removeHs=False)
mol1 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol1)

mol2 = Chem.MolFromPDBFile(pdb2_path, sanitize=True, removeHs=False)
mol2 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol2)

mol3 = Chem.MolFromPDBFile(pdb3_path, sanitize=True, removeHs=False)
mol3 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol3)

mol4 = Chem.MolFromPDBFile(pdb4_path, sanitize=True, removeHs=False)
mol4 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol4)

mol5 = Chem.MolFromPDBFile(pdb5_path, sanitize=True, removeHs=False)
mol5 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol5)

mol6 = Chem.MolFromPDBFile(pdb6_path, sanitize=True, removeHs=False)
mol6 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol6)

mol7 = Chem.MolFromPDBFile(pdb7_path, sanitize=True, removeHs=False)
mol7 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol7)

mol9 = Chem.MolFromPDBFile(pdb9_path, sanitize=True, removeHs=False)
mol9 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol9)

mol8 = Chem.MolFromPDBFile(pdb8_path, sanitize=True, removeHs=False)
mol8 = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol8)

out_mol = copy.deepcopy(mol1)
out_mol.RemoveAllConformers()

for i in range(increment):
    out_mol.AddConformer(mol1.GetConformer(i))
for i in range(increment):
    out_mol.AddConformer(mol2.GetConformer(i + increment))
for i in range(increment):
    out_mol.AddConformer(mol3.GetConformer(i + 2*increment))
for i in range(increment):
    out_mol.AddConformer(mol4.GetConformer(i + 3*increment))
for i in range(increment):
    out_mol.AddConformer(mol5.GetConformer(i + 4*increment))
for i in range(increment):
    out_mol.AddConformer(mol6.GetConformer(i + 5*increment))
for i in range(increment):
    out_mol.AddConformer(mol7.GetConformer(i + 6*increment))
for i in range(increment):
    out_mol.AddConformer(mol8.GetConformer(i + 7*increment))
for i in range(increment):
    out_mol.AddConformer(mol9.GetConformer(i + 8*increment))

Chem.MolToPDBFile(out_mol, out_name)
