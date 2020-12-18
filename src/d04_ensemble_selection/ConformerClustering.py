from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import pickle
from pathlib import Path
from src.d04_ensemble_selection.clustering_utils import get_cluster_centers, cluster_threshold_plot

fname = "CsE_ccl4_corr_noebounds_5400"
mol = Chem.MolFromPDBFile("../../data/02_CsE/21_CsE_ccl4_corr_noebounds_5400.pdb", removeHs=False)


# RMS matrix: Check if matrix already exists. Generation can take a long time!
if Path("../../data/99_tmp/rmsmat_" + fname).exists():
    rmsmat = pickle.load(open("../../data/99_tmp/rmsmat_" + fname, "rb"))
else:
    rmsmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
    pickle.dump(rmsmat, open("../../data/99_tmp/rmsmat_" + fname, "wb"))


num = mol.GetNumConformers()

cluster_threshold_plot(rmsmat, num, 0.5, 6, 0.5)

index = get_cluster_centers(rmsmat, num, 3.5)
print(index[:10])
