import numpy as np
from scipy.optimize import fmin_slsqp

CDM = np.array([[0.15, 0.2, 0.25], [0.25, 0.1, 0.25], [0.05, 0.25, 0.4], [0.1, 0.3, 0.35]]) # matrix of n conformers and their m distances (m*n matrix)
CDM = np.transpose(CDM)
N = np.array([0.2, 0.15, 0.25]) # NOE distance restraints (length m)
N = np.transpose(N)
tol = 0.1 # equivalent of experimental uncertainty

w0 = np.array([0.5, 0.5, 0., 0.]) # guess weight fractions
_, m = np.shape(CDM)
w0 = np.ones(m)/m
w0 = np.transpose(w0)


# setup of the equality constraint
def normality(x):
    '''make sure weigths sum to 1'''
    return (1 - np.sum(x))

# setup of the inquality constraints. Naturally, these would be
# formulated as vector equations. Since the minimizer
# requires scalar functions, they are generated on the fly.
pos = []
lb = []
ub = []

for i in range(len(w0)):
    def positivity(x, i=i):
        return x[i]
    pos.append(positivity)

for i in range(len(CDM)):
    def lower_bounds(x, i=i):
        return CDM[i] * x[i] - (1 - tol) * N[i]
    lb.append(lower_bounds)

    def upper_bound(x, i=i):
        return (1 + tol) * N[i] - CDM[i] * x[i]
    ub.append(upper_bound)


def objective(x):
    # returns the RMSD between prediction and actual value
    return np.sum(np.square(np.matmul(CDM, w0) - N))/len(N)



test = np.matmul(CDM, w0)
bounds= pos + lb + ub
#tt = zip(normality)
w, f_obj_val, its, exit_mode, s_mode = fmin_slsqp(objective, w0, eqcons=[normality], ieqcons=bounds,
                                                  acc=1e-6, full_output=True)

#evaluate
print("Weights:")
print(w)

print("Actual distances:")
print(N)

print("Pred distances:")
print(np.matmul(CDM, w))