from src.d04_ensemble_selection.MolDistances import get_distances, get_distances_new
import time
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from src.d04_ensemble_selection.NAMFIS_utils import namfis, namfis_cobyla
from src.d04_ensemble_selection.clustering_utils import get_cluster_centers
import os

t = time.time()

pdb_code = '0CSA'
fname = "CsA_chcl3_noebounds_5400"

os.chdir('../../data/' + pdb_code.upper())
mol = Chem.MolFromPDBFile("21_0csa_tol1_noebounds_5400.pdb", removeHs=False)
# mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf294.pdb", removeHs=False)
noe_df = pd.read_csv("10_CsA_chcl3_noebounds.csv", sep="\s", comment="#")
noe_index = pd.read_csv(f'10_{pdb_code.lower()}_NOE_index.csv', index_col=0)

N, CDM = get_distances_new(mol, noe_index)

# clustering: Get rmsmat from ConformerClustering.py
rmsmat = pickle.load(open("../99_tmp/rmsmat_" + fname, "rb"))
num = mol.GetNumConformers()
index = get_cluster_centers(rmsmat, num, 4.5)
print(f"Clustering of {num} conformers resulted in {len(index)} being chosen.")

N = N[:].to_numpy()
CDM = np.transpose(CDM)
CDM = CDM[:,index]

w, f_obj_val, its, exit_mode, low_mode = namfis(CDM=CDM, NOE=N, rand=True, seed=1, tol=0.1,
                                                max_runs=200, max_iter=10*000)
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

print("Actual - Pred distances:")
print(np.around(np.matmul(CDM, w)[:20] - N[:20], 2))

runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))
