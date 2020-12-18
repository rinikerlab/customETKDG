import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from src.d04_ensemble_selection.MolDistances import get_distances, get_distances_new
from src.d04_ensemble_selection.clustering_utils import get_cluster_centers
import matplotlib.pyplot as plt
import time
import os
import pickle

plt.style.use('seaborn-white')

t = time.time()

pdb_code = '0CSA'
fname = "CsA_chcl3_noebounds_5400"

os.chdir('../../data/' + pdb_code.upper())
mol = Chem.MolFromPDBFile("21_0csa_tol1_noebounds_5400.pdb", removeHs=False)
noe_index = pd.read_csv(f'10_{pdb_code.lower()}_NOE_index.csv', index_col=0)

N, CDM = get_distances_new(mol, noe_index)

# clustering: Get rmsmat from ConformerClustering.py
rmsmat = pickle.load(open("../99_tmp/rmsmat_" + fname, "rb"))
num = mol.GetNumConformers()
index = get_cluster_centers(rmsmat, num, 3.9)
print(f"Clustering of {num} conformers resulted in {len(index)} being chosen.")

N = N[:].to_numpy()
CDM = np.transpose(CDM)
CDM = CDM[:,index]

X = CDM
y_true = N.reshape(-1, 1)

loss = "squared_epsilon_insensitive"
loss = "squared_loss"
reg = SGDRegressor(loss=loss, penalty="l1", alpha=0.01, epsilon=1, max_iter=100000,
                         tol=1e-3, learning_rate="invscaling", shuffle=True, random_state=42, fit_intercept=False)
#reg = make_pipeline(StandardScaler(), regressor)
reg.fit(X, y_true)

"""
params = reg.get_params()
print(params)
print(reg.score(X, y))
print(regressor.coef_)
print(regressor.intercept_)
"""

y_pred = reg.predict(X)
y_pred = y_pred.reshape(-1, 1)
# The coefficients
weights = reg.coef_
weights[weights < 0] = 0
weights = weights/np.sum(weights)
reg.coef_ = weights
weights = weights.reshape(-1, 1)
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_true, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: {:.2f}'.format(r2_score(y_true, y_pred)))


my_pred = np.matmul(CDM, weights)
# Plot outputs
plt.scatter(y_pred, y_true, color='black')
#plt.plot(X[:,2], y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

y_pred = my_pred
y_pred = np.transpose(y_pred)[0]
y_true = np.transpose(y_true)[0]
weights = np.transpose(weights)[0]



print(f"Weights (sum is {np.sum(weights)}):")
print(np.sort(weights)[-10:])

print("Actual distances:")
print(np.around(y_pred[:20], 2))

print("Pred distances:")
print(np.around(y_pred[:20], 2))

print("Pred - Actual distances:")
print(np.around(y_pred[:20] - y_true[:20], 2))

print("Summed violations:")
tmp = y_pred - y_true
tmp[tmp < 0] = 0
print(np.sum(tmp))

print("Summed violations per bond:")
print(np.sum(tmp)/len(y_true))