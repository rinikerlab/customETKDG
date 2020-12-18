import time
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import tqdm
import copy


def get_noe_pair_dist(rdmol, noe_index):
    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(rdmol, i) for i in tqdm.tqdm(range(rdmol.GetNumConformers()))])
    distances = distance_matrix_for_each_conformer[:, noe_index['Atm_idx_1'], noe_index['Atm_idx_2']]
    return distances


def noe_violations(ref_dist, meth_pair_dist, dev_tol=0):
    pos_violations = []
    difference = []
    for i in range(len(meth_pair_dist)):
        tmp = (meth_pair_dist[i] - ref_dist) - dev_tol
        difference.append(copy.deepcopy(tmp))
        tmp[tmp < 0] = 0
        pos_violations.append(tmp)

    sum_viol = []
    for i in range(len(pos_violations)):
        sum_viol.append(np.sum(pos_violations[i]))

    return pos_violations, difference, sum_viol

# Starting from a (hopefully) reasonable starting conformer, the algorithm greedily adds a new complementary conformer
# and then determines the best mixing ratio between old and new conf. It then repeats this until (adding new confs
# to existing set and reweighting) until convergence or max iters.


t = time.time()

pdb_code = '0csa'

solventDielectric = 4.8 #TODO
pdb_path = f'21_{pdb_code}_tol1_noebounds_5400.pdb'
rdmol_df_path = f'10_{pdb_code}_NOE_index.csv'
os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')

smiles_mol = Chem.MolFromSmiles(smiles)
rdmol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=False)
rdmol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, rdmol)
rdmol_df = pd.read_csv(rdmol_df_path, index_col=0)

ref_dist = rdmol_df['Dist_[A]']
noe_pair_dist = get_noe_pair_dist(rdmol, rdmol_df)

rel_pair_dist = copy.deepcopy(noe_pair_dist)
for i in range(len(noe_pair_dist)):
    rel_pair_dist[i] = rel_pair_dist[i] - ref_dist

pos_violations, difference, sum_viol = noe_violations(ref_dist, noe_pair_dist, dev_tol=0)
best_index = np.argsort(sum_viol)
start_idx = best_index[0]
new_idx = best_index[1:]

indexes = copy.deepcopy(new_idx)
ensemble_idx = [start_idx]
ensemble_wgts = np.array(1)

print(f'Best summed violation: {sum_viol[start_idx]}')

pow = -6  # NOE averaging, might be -3 in some cases
cur_distances = noe_pair_dist[start_idx]  # initialize
anti_tol = 0.01  # aim for slightly more compact average than the upper bound suggests

# TODO: Consider also penalizing a too compact ensemble (lower bounds would be valuable...)
# TODO: How much below an upper bound is too compact???
for x in range(100):  # 100 iterations should suffice for convergence (if it's even possible),
                      # and we do not want bigger ensembles anyways
    print(f'# {x} ###########################################')
    bval = 0  # TODO: treat stopping case
    for i in indexes:
        tmp_pos_viol = cur_distances - ref_dist + anti_tol
        tmp_pos_viol[tmp_pos_viol < 0] = 0
        newval = np.dot(tmp_pos_viol, difference[i])
        if newval < bval:
            new_idx = i
            bval = newval

    if bval == 0:  # TODO: Test if this is too harsh
        print(f'Scalar ({bval}) indicates no further improvement. Stopping iterating.')
        break
    print(f'Scalar {bval} for conformer at idx {new_idx}')

    bweight = float('inf')
    for i in np.linspace(1, 0, 101):  # TODO: An iterative solver would be better than grid searching
        weighted_diff = np.power(i * np.power(cur_distances, pow) + (1 - i) * np.power(noe_pair_dist[new_idx], pow), 1 / pow) - ref_dist
        tmp = copy.deepcopy(weighted_diff)
        tmp = tmp + anti_tol
        tmp[tmp < 0] = 0
        sweight = np.sum(tmp)
        if sweight < bweight:
            bweight = sweight
            b_i = i
            new_distances = weighted_diff + ref_dist

    if b_i == 1:  # Necessary if anti-tol demands too much / complete fulfillment is not possible
        print(f'Rescaling of ({b_i}) indicates no further improvement, some bounds still violated. Stopping iterating.')
        break

    print(f'Best combined violation: {np.round(bweight, 4)} with previous population rescaled to {np.round(b_i, 4)}')
    indexes = np.delete(indexes, new_idx)  # remove already chosen conformer TODO: Consider not removing it, probably not much of a difference
    cur_distances = new_distances
    ensemble_idx.append(new_idx)
    ensemble_wgts = np.append(b_i*ensemble_wgts, 1 - b_i)

assert len(ensemble_idx) == len(ensemble_wgts)

print(f'Selected {len(ensemble_idx)} conformers for ensemble.')
print(ensemble_idx)
print(np.round(ensemble_wgts, 3))
print(f'Normality check: {np.sum(ensemble_wgts)}')
print(f'Distances: {cur_distances}')
print(f'Dist - NOE_dist: {cur_distances-ref_dist}')

runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))