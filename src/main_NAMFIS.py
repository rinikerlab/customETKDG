from MolDistances import get_distances
import time
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from NAMFIS_utils import namfis, namfis_cobyla
from _ConformerClustering import get_cluster_centers

t = time.time()

mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb", removeHs=False)
# mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf294.pdb", removeHs=False)
noe_df = pd.read_csv("/home/kkajo/Workspace/Git/MacrocycleConfGenNOE/src/CsA/CsA_chcl3_noebounds.csv", sep="\s",
                     comment="#")

N, CDM = get_distances(mol, noe_df)

# clustering: Get rmsmat from ConformerClustering.py
fname = "CsA_chcl3_noebounds_numconf5400"
rmsmat = pickle.load(open("Data/rmsmat_" + fname, "rb"))
num = mol.GetNumConformers()
index = get_cluster_centers(rmsmat, num, 3.5)
print(f"Clustering of {num} conformers resulted in {len(index)} being chosen.")

N = N[:]
CDM = CDM[:,index]

w, f_obj_val, its, exit_mode, low_mode = namfis(CDM=CDM, NOE=N, rand=True, seed=1, tol=0.4,
                                                max_runs=20, max_iter=1000)
#w, f_obj_val = namfis_cobyla(CDM=CDM, NOE=N, rand=False, seed=1, tol=0.3,
#                                                max_runs=200, max_iter=int(1e16))

# evaluate
print(f"Made {its} iterations.")
print(f"Best objective: {f_obj_val}")
print(f"Exit mode: {exit_mode}")
print(f"Lowest exit mode: {low_mode}")

print(f"Weights (sum is {np.sum(w)}):")
print(np.sort(w)[-10:])

print("Actual distances:")
print(N[:20])

print("Pred distances:")
print(np.around(np.matmul(CDM, w), 2)[:20])

runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))
