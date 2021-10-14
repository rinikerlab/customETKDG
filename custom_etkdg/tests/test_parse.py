import pytest

from custom_etkdg.utils import *
from custom_etkdg.mol_ops import *
from custom_etkdg.noe import NOE
from rdkit import Chem
import logging

logging.getLogger().setLevel(logging.INFO)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

smiles = open(get_data_file_path("6VY8/6vy8.smi"), "r").readlines()[0].strip()
smiles = "CC[C@H](C)[C@@H]1NC(=O)[C@@H]2CCCN2C(=O)[C@@H]3CCCN3C(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@@H](NC(=O)[C@@H]4CN5N=NC=C5C[C@H](NC1=O)C(=O)N[C@@H](Cc6ccccc6)C(=O)N7CCC[C@H]7C(=O)N[C@@H](CC([O-])=O)C(=O)NCC(=O)N[C@@H](CCCNC(N)=[NH2+])C(=O)N4)[C@@H](C)O)[C@@H](C)CC"
mol = mol_from_smiles_pdb(smiles, get_data_file_path("6VY8/6vy8.pdb"), infer_names = True)


# mol = rename(mol, "O", "XX")

# print(Chem.MolToPDBBlock(mol))


noe = NOE()
noe.from_explor(get_data_file_path("6VY8/6vy8.mr"))
print(type(noe))

# print(noe.__dict__)
print(noe.add_noe_to_mol(mol))