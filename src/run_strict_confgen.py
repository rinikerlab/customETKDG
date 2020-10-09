import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
from _utils import assign_hydrogen_pdbinfo, get_noe_restraint_bmat
import time

"""
# give NOE file as first parameter, ref.pdb and ref.smi files in same directory
# number of conformers as second parameter
"""

t = time.time()

try:
    print("NOE file path: {}".format(sys.argv[1]))
except:
    print("Must specify path to NOE file.")
    exit()

try:
    num_conf = int(sys.argv[2])
    print("Number of conformers: {}".format(sys.argv[2]))
except:
    num_conf = 5400
    print("Defaulting number of conformers to {}".format(num_conf))

input_path = os.path.dirname(sys.argv[1])
#print(input_path)

with open("{}/ref.smi".format(input_path), "r") as tmp:
    smiles = tmp.read().replace("\n", "")
ref_pdb_path = "{}/ref.pdb".format(input_path)

smiles_mol = Chem.MolFromSmiles(smiles)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)


mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))
#print(Chem.MolToPDBBlock(mol))

# read NOE data
df = pd.read_csv(sys.argv[1], sep = "\s", comment = "#")


bmat = get_noe_restraint_bmat(mol, df)        
        
params = AllChem.ETKDG()  # don't explicitly specify v3 bc to be run in Docker
#params = AllChem.ETKDGv3()
params.useRandomCoords = False
#params.SetBoundsMat(bmat)     # not available on Docker, see below
params.randomSeed = 42
params.verbose = False
params.numThreads = 0           # use parallelism

#AllChem.EmbedMultipleConfs(mol, num_conf, params)
AllChem.EmbedMultipleConfs(mol, num_conf, params, boundsMatrix = bmat)  # for Docker
Chem.MolToPDBFile(mol, "{}_numconf{}.pdb".format(sys.argv[1][:-4], num_conf))

runtime = time.time() - t
print("Took {:.0f} s to generate {} conformers.".format(runtime, num_conf))