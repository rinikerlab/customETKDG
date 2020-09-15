import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
from _utils import assign_hydrogen_pdbinfo, get_noe_restraint_bmat

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


df = pd.read_csv(sys.argv[1], sep = "\s", comment = "#")


bmat = get_noe_restraint_bmat(mol, df)        
        
params = AllChem.ETKDGv3()
params.useRandomCoords = False #TODO changeable

params.SetBoundsMat(bmat)

AllChem.EmbedMultipleConfs(mol, num_conf , params)
Chem.MolToPDBFile(mol, "{}_numconf{}.pdb".format(sys.argv[1][:-4], num_conf))
