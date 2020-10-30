from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina
import pickle
import numpy as np
import matplotlib.pyplot as plt
from _ConformerClustering import get_cluster_centers, cluster_threshold_plot

fname = "CsE_ccl4_corr_noebounds_numconf5400"
mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsE/CsE_ccl4_corr_noebounds_numconf5400.pdb", removeHs=False)


# RMS matrix: Uncomment this code if the matrix does not exist yet. Can take a long time!
"""
rmsmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
pickle.dump(rmsmat, open("Data/rmsmat_" + fname, "wb"))
"""
rmsmat = pickle.load(open("Data/rmsmat_" + fname, "rb"))

num = mol.GetNumConformers()

cluster_threshold_plot(rmsmat, num, 0.5, 6, 0.5)

index = get_cluster_centers(rmsmat, num, 3.5)
print(index[:10])

print("done")
