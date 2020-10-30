from LassoRegression import coordinate_descent_lasso
import numpy as np
import pandas as pd
from rdkit import Chem
import sys
from sklearn import datasets
from matplotlib import pyplot as plt
from MolDistances import get_distances
import time
from _ConformerClustering import get_cluster_centers

plt.style.use('seaborn-white')

t = time.time()

mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb", removeHs=False)
#mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf294.pdb", removeHs=False)
noe_df = pd.read_csv("/home/kkajo/Workspace/Git/MacrocycleConfGenNOE/src/CsA/CsA_chcl3_noebounds.csv", sep = "\s", comment = "#")

y_true, X = get_distances(mol, noe_df)
y_true = y_true.reshape(-1, 1)

# Initialize variables
m, n = X.shape
initial_theta = np.ones((n, 1))/n # start with equal weights, normalized
#initial_theta = np.zeros((n, 1)) # start with zeros -- no sig difference
#initial_theta = np.random.rand((n, 1))
theta_list = list()
y_pred = []
dy = []
alphas = [6] #, 0.1, 10, 100] #np.logspace(0, 4, 2) / 10  # Range of alpha regularization values
#alphas = [20, 40, 50, 60, 70, 100]
#alphas = [1e-5, 1e-4, 1e-3, 1e-2]
#alphas = [1]

# Run lasso regression for each lambda
for l in alphas:
    print(f"--- Now running for alpha={l}. ---")
    theta, y_p = coordinate_descent_lasso(initial_theta, X, y_true, alpha=l, num_iters=5000, intercept=False)
    y_pred.append(y_p)
    dy.append(y_p - y_true)
    theta = theta / (np.sum(theta) + sys.float_info.epsilon) # normalize final weights
    theta_list.append(theta)

# Stack into numpy array
theta_lasso = np.stack(theta_list).T
y_pred = np.stack(y_pred).reshape(-1, m).T
dy = np.stack(dy).reshape(-1, m).T

neg_count = len(list(filter(lambda x: (x < 0), dy)))
pos_count = len(list(filter(lambda x: (x >= 0), dy)))
print("Positive numbers : ", pos_count)
print("Negative numbers : ", neg_count)

# save for inspection
df = pd.DataFrame(theta_lasso)
df.to_csv("thetas.csv", header=alphas)

# Plot results
n, _ = theta_lasso.shape
plt.figure(figsize=(12, 8))

# do not plot zeros
#theta_lasso[theta_lasso == 0] = np.nan

for i in range(n):
    plt.plot(alphas, theta_lasso[i], label=np.arange(1, n))

plt.xscale('log') # alt: "log" "linear"
plt.xlabel('Log($\\lambda$)')
plt.ylabel('Coefficients')
plt.title('Lasso Paths - Numpy implementation')
#plt.legend()
plt.axis('tight')
plt.savefig("test.png")

runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))