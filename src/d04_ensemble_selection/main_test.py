from LassoRegression import coordinate_descent_lasso

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

plt.style.use('seaborn-white')

# Load the diabetes dataset. In this case we will not be using a constant intercept feature
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)

# Initialize variables
m, n = X.shape
initial_theta = np.ones((n, 1))
theta_list = list()
alphas = np.logspace(0, 4, 300) / 10  # Range of alpha regularization values

# Run lasso regression for each lambda
for l in alphas:
    theta = coordinate_descent_lasso(initial_theta, X, y, alpha=l, num_iters=100, intercept=False)
    theta_list.append(theta)

# Stack into numpy array
theta_lasso = np.stack(theta_list).T

# Plot results
n, _ = theta_lasso.shape
plt.figure(figsize=(12, 8))

for i in range(n):
    plt.plot(alphas, theta_lasso[i], label=diabetes.feature_names[i])

plt.xscale('log')
plt.xlabel('Log($\\lambda$)')
plt.ylabel('Coefficients')
plt.title('Lasso Paths - Numpy implementation')
plt.legend()
plt.axis('tight')
plt.savefig("test.png")
