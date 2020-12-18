from src.d00_utils.utils import *
import time
import copy

# Just a short script to combine pdb models from different simulations.
# Probably of no further use.

t = time.time()

increment = 18

pdb_code = '6fce'
os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'

pdb1_path = f'42_5ns_vac_0-17idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb2_path = f'42_5ns_vac_18-35idx_{pdb_code}_tol1_noebounds_5400.pdb'
pdb3_path = f'42_5ns_vac_36-53idx_{pdb_code}_tol1_noebounds_5400.pdb'

#pdb1_path = f'52_5ns_solv_0-17idx_{pdb_code}_tol1_noebounds_5400.pdb'
#pdb2_path = f'52_5ns_solv_18-35idx_{pdb_code}_tol1_noebounds_5400.pdb'
#pdb3_path = f'52_5ns_solv_36-53idx_{pdb_code}_tol1_noebounds_5400.pdb'

out_name = f'42_5ns_vac_{pdb_code}_tol1_noebounds_54.pdb'
#out_name = f'52_5ns_solv_{pdb_code}_tol1_noebounds_54.pdb'

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

out_mol = copy.deepcopy(mol1)
out_mol.RemoveAllConformers()

for i in range(increment):
    out_mol.AddConformer(mol1.GetConformer(i))
for i in range(increment):
    out_mol.AddConformer(mol2.GetConformer(i + increment))
for i in range(increment):
    out_mol.AddConformer(mol3.GetConformer(i + 2*increment))

Chem.MolToPDBFile(out_mol, out_name)
