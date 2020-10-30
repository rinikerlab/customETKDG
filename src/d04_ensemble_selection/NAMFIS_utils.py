import numpy as np
from scipy.optimize import fmin_slsqp, fmin_cobyla
from joblib import Parallel, delayed


def namfis(CDM, NOE, w0 = None, tol: float = 0.1, rand: bool = False, seed: int = None, max_runs: int = 1,
           max_iter: int = 1000):
    """
    Function to calculate NAMFIS weights for an ensemble of conformers wrt a set of NOE restraints.
    :param CDM: Conformer-Distance Matrix. Each of the m rows represents a NOE distance pair, each of
    the n colums a conformer.
    :param NOE: A vector of m experimental NOE restraints. Ordering must correspond to the one in CDM.
    :param w0: Initial weight vector for optimization of length n. If left blank, it will be initialized uniformly if
    rand = False and randomly otherwise. Should be normalized.
    :param tol: Tolerance factor for upper and lower bound wrt the experimental NOE values. Tol=0.1 means +/- 10%.
    :param rand: If weights should be initialized uniformly (False) or randomly. If True, the function can optimized in
    parallel with different starting points.
    :param max_runs: Number of newly initialized optimization to be performed in parallel. Has only an effect if
    rand = True
    :param max_iter: Maximum number of iterations per optimization (passed on to optimizer).
    :return: the final weight vector best_w, the best value of the objective best_obj, total number of iterations
    tot_its, the final exit mode ex_mode, and the lowest exit mode low_mode
    """

    _, m = np.shape(CDM)

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

    for i in range(m):
        def positivity(x, i=i):
            return x[i]

        pos.append(positivity)

    for i in range(len(CDM)):
        def lower_bounds(x, i=i):
            return np.matmul(CDM[i], x) - (1 - tol) * NOE[i]

        lb.append(lower_bounds)

        def upper_bound(x, i=i):
            return (1 + tol) * NOE[i] - np.matmul(CDM[i], x)

        ub.append(upper_bound)

    bounds = pos + lb + ub
    #bounds = pos

    def objective(x):
        # returns the RMSD between prediction and actual value
        return np.sum(np.square(np.matmul(CDM, x) - NOE)) / len(NOE) # penalize all deviations
        #return np.sum(np.square(np.min(NOE - np.matmul(CDM, x), 0))) / len(NOE) # penalize only overshoot


    if rand:
        if (seed is not None):
            np.random.seed(seed)

        best_w = None
        best_obj = np.inf
        tot_its = 0
        ex_mode = 10
        low_mode = 10


        wtot = []
        for i in range(max_runs):
            w0 = np.random.rand(m)
            w0 = w0/np.sum(w0)
            wtot.append(w0)

        results = Parallel(n_jobs=8)(delayed(fmin_slsqp)(objective, wtot[i], eqcons=[normality], ieqcons=bounds, acc=1e-12, iter=max_iter, full_output=True, iprint=0) for i in range(max_runs))
        #w, f_obj_val, its, exit_mode, s_mode = fmin_slsqp(objective, w0, eqcons=[normality], ieqcons=bounds, acc=1e-6, iter=max_iter, full_output=True, iprint=0)
            #if (f_obj_val < best_obj):
                #best_w = w
        #best_obj[i] = f_obj_val
                #ex_mode = exit_mode
            #if (ex_mode < low_mode):
                #low_mode = ex_mode
            #tot_its = tot_its + its
        #best_obj = np.min(best_obj)
        for i in range(max_runs):
            if (results[i][1] < best_obj):
                best_w = results[i][0]
                best_obj = results[i][1]
                ex_mode = results[i][3]
            if (ex_mode < low_mode):
                low_mode = ex_mode
            tot_its = tot_its + results[i][2]

        return best_w, best_obj, tot_its, ex_mode, low_mode

    else:
        if w0 is None:
            w0 = np.ones(m) / m

        w, f_obj_val, its, exit_mode, s_mode = fmin_slsqp(objective, w0, eqcons=[normality], ieqcons=bounds,
                                                          acc=1e-12, iter=max_iter, full_output=True, iprint=2)
        #w = fmin_cobyla(objective, w0, cons=bounds, disp=3, maxfun=max_iter)
        #f_obj_val = 0
        #its= 0
        #exit_mode = 0
        return w, f_obj_val, its, exit_mode, exit_mode



def namfis_cobyla(CDM, NOE, w0 = None, tol: float = 0.1, rand: bool = False, seed: int = None, max_runs: int = 1, max_iter: int = 1000):
    """

    :param CDM:
    :param NOE:
    :param w0:
    :param tol:
    :param rand:
    :param max_runs:
    :param max_iter:
    :return:
    """

    _, m = np.shape(CDM)

    # setup of the equality constraint
    def normality(x):
        '''make sure weigths sum to 1'''
        return (1 - np.sum(x))

    def neg_normality(x):
        '''make sure weigths sum to 1'''
        return -(1 - np.sum(x))

    # setup of the inquality constraints. Naturally, these would be
    # formulated as vector equations. Since the minimizer
    # requires scalar functions, they are generated on the fly.
    pos = []
    lb = []
    ub = []

    for i in range(m):
        def positivity(x, i=i):
            return x[i]

        pos.append(positivity)

    for i in range(len(CDM)):
        def lower_bounds(x, i=i):
            return np.matmul(CDM[i], x) - (1 - tol) * NOE[i]

        lb.append(lower_bounds)

        def upper_bound(x, i=i):
            return (1 + tol) * NOE[i] - np.matmul(CDM[i], x)

        ub.append(upper_bound)

    bounds = pos + lb + ub + [normality] + [neg_normality]
    #bounds = pos

    def objective(x):
        # returns the RMSD between prediction and actual value
        #return 1000*np.square(1 - np.sum(x)) + np.sum(np.square(np.matmul(CDM, x) - NOE)) / len(NOE) # penalize all deviations
        #return np.sum(np.square(np.min(NOE - np.matmul(CDM, x), 0))) / len(NOE) # penalize only overshoot
        return np.sum(np.square(np.matmul(CDM, x) - NOE)) / len(NOE)


    if rand:
        if (seed is not None):
            np.random.seed(seed)

        best_w = None
        best_obj = np.inf
        wtot = []

        for i in range(max_runs):
            w0 = np.random.rand(m)
            w0 = w0/np.sum(w0)
            wtot.append(w0)

        results = Parallel(n_jobs=8)(delayed(fmin_cobyla)(objective, wtot[i], cons=bounds, maxfun=max_iter, disp=0) for i in range(max_runs))
        #w, f_obj_val, its, exit_mode, s_mode = fmin_slsqp(objective, w0, eqcons=[normality], ieqcons=bounds, acc=1e-6, iter=max_iter, full_output=True, iprint=0)
            #if (f_obj_val < best_obj):
                #best_w = w
        #best_obj[i] = f_obj_val
                #ex_mode = exit_mode
            #if (ex_mode < low_mode):
                #low_mode = ex_mode
            #tot_its = tot_its + its
        #best_obj = np.min(best_obj)
        for i in range(max_runs):
            obj = objective(results[i])
            if (obj < best_obj):
                best_w = results[i]
                best_obj = obj
        return best_w, best_obj

    else:
        if w0 is None:
            w0 = np.ones(m) / m

        w = fmin_cobyla(objective, w0, cons=bounds, maxfun=max_iter, disp=3)
        obj = objective(w)
        #f_obj_val = 0
        #its= 0
        #exit_mode = 0
        return w, obj