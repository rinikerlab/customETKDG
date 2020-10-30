import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from MolDistances import get_distances
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-white')

t = time.time()

mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb", removeHs=False)
noe_df = pd.read_csv("/home/kkajo/Workspace/Git/MacrocycleConfGenNOE/src/CsA/CsA_chcl3_noebounds.csv", sep = "\s", comment = "#")

y_true, X = get_distances(mol, noe_df)
y_true = y_true.reshape(-1, 1)

loss = "squared_epsilon_insensitive"
loss = "squared_loss"
reg = SGDRegressor(loss=loss, penalty="l1", alpha=0.01, epsilon=1, max_iter=10000,
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
weights = weights/np.sum(weights)
weights = weights.reshape(-1, 1)
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_true, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: {:.2f}'.format(r2_score(y_true, y_pred)))

# Plot outputs
plt.scatter(X, y_true, color='black')
#plt.plot(X[:,2], y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()