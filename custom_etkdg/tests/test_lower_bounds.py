import pytest

from custom_etkdg.utils import *
from custom_etkdg.mol_ops import *
from custom_etkdg.noe import NOE
from rdkit import Chem
import logging

logging.getLogger().setLevel(logging.INFO)

smiles = open(get_data_file_path("6BF3/6bf3.smi"), "r").readlines()[0].strip()
mol = mol_from_smiles_pdb(smiles, get_data_file_path("6BF3/6bf3.pdb"), infer_names = True)


noe = NOE()
noe.from_explor(get_data_file_path("6BF3/6bf3.mr"))


resmol = noe.add_noe_to_mol(mol)
resmol.update_bmat()
assert resmol.bmat[25][86]==4.7
assert resmol.bmat[86][25]==2.4
resmol.scale_upper_distances(1.4)
resmol.update_bmat() #!!!! always need to update the bmat after changing the scaling
assert resmol.bmat[25,86]==4.7*1.4
assert resmol.bmat[86,25]==2.4

resmol = noe.add_noe_to_mol(mol)
resmol.update_bmat(include_lower_bounds=True)
assert resmol.bmat[25,86]==4.7
assert resmol.bmat[86,25]==2.3
resmol.scale_lower_distances(1.4)
resmol.update_bmat(include_lower_bounds=True) #!!!! always need to update the bmat after changing the scaling
assert resmol.bmat[25,86]==4.7
assert resmol.bmat[86,25]==2.3*1.4



