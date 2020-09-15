import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
from _utils import assign_hydrogen_pdbinfo

try:
    num_conf = int(sys.argv[2])
except:
    num_conf = 5400
  
input_path = os.path.dirname(sys.argv[1])
with open("{}/ref.smi".format(input_path), "r") as tmp:
    smiles = tmp.read().replace("\n", "")

ref_pdb_path = "{}/ref.pdb".format(input_path)

smiles_mol = Chem.MolFromSmiles(smiles)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)

    

mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))
#print(Chem.MolToPDBBlock(mol))

        
params = AllChem.ETKDGv3()
params.useRandomCoords = True #TODO changeable
params.useMacrocycle14config = True
#params.useMacrocycleTorsions = True
params.verbose = True
print(params.__dict__)


AllChem.EmbedMultipleConfs(mol, num_conf , params)
Chem.MolToPDBFile(mol, "{}_numconf{}.pdb".format(sys.argv[1][:-4], num_conf))
