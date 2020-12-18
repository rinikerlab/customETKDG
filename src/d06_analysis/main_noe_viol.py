from src.d06_analysis.analysis_utils import *
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
import sys
import os
from src.d00_utils.utils import *
import time
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import pickle
import matplotlib
import matplotlib.pyplot as plt


'''
dict = {
    'name': sys,
    'smiles': smiles,
    'solvent': get_solvent(sys),
    'dielectric': get_dielectric(sys),
    'noe_index': noe_index,
    'ref': ref,
    'bl_etkdg': bl,
    'noe_etkdg': noe_etkdg,
    'omega': omega,
    'vac_min': vac_min,
    'vac_1ns': vac_1ns,
    'vac_5ns': vac_5ns,
    'solv_min': solv_min,
    'ANI_min': ANI_min
    
    bond_dist
    
    'noe_pair_dist'
    bl_pair_dist
    omega_pair_dist
    vac_min_pair_dist
    solv_min_pair_dist
    ANI_min_pair_dist
    
    noe_rb_rmsd
    bl_rb_rmsd
    #omega_rb_rmsd
    vac_min_rb_rmsd
    solv_min_rb_rmsd
    ANI_min_rb_rmsd
    
    noe_tfd
    bl_tfd
    vac_min_tfd
    solv_min_tfd
    ANI_min_tfd
}
'''

os.chdir('../../results/')
data = pickle.load(open('../data/analysis_notfd_gene.data', 'rb'))
'''
for sys in data:
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['bl_pair_dist'], sys['name']+'normal')
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['vac_min_pair_dist'],
                       sys['name'] + 'vac_min')
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['ANI_min_pair_dist'],
                       sys['name'] + 'ANI_min')
'''

#data = data[:2]

tol = 0
best_noe = []
ten_best_noe = []
bl_viols = []
noe_viols = []
vac_min_viols = []
vac_1ns_viols = []
vac_5ns_viols = []
solv_min_viols = []
solv_1ns_viols = []
solv_5ns_viols = []
ANI_min_viols = []
names = []
err = -0.05

for sys in data:
    print(sys['name'])
    names.append(sys['name'])

    _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['bl_pair_dist'], dev_tol=tol)
    bl_viols.append(min(sum_viol)/len(sys['noe_index']['Dist_[A]']))  # normalize

    violations, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['noe_pair_dist'], dev_tol=tol)
    noe_viols.append(min(sum_viol)/len(sys['noe_index']['Dist_[A]']))
    best_noe.append(violations[sum_viol.index(min(sum_viol))])
    ind = np.argsort(sum_viol)[:20]
    arr = []
    for n in ind:
        arr.append(violations[n])
    ten_best_noe.append(arr)

    _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['vac_min_pair_dist'], dev_tol=tol)
    vac_min_viols.append(min(sum_viol)/len(sys['noe_index']['Dist_[A]']))

    try:
        _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['vac_1ns_pair_dist'], dev_tol=tol)
        vac_1ns_viols.append(min(sum_viol) / len(sys['noe_index']['Dist_[A]']))
    except:
        vac_1ns_viols.append(err)

    try:
        _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['vac_5ns_pair_dist'], dev_tol=tol)
        vac_5ns_viols.append(min(sum_viol) / len(sys['noe_index']['Dist_[A]']))
    except:
        vac_5ns_viols.append(err)

    _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['solv_min_pair_dist'], dev_tol=tol)
    solv_min_viols.append(min(sum_viol)/len(sys['noe_index']['Dist_[A]']))

    try:
        _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['solv_1ns_pair_dist'], dev_tol=tol)
        solv_1ns_viols.append(min(sum_viol) / len(sys['noe_index']['Dist_[A]']))
    except:
        solv_1ns_viols.append(err)

    try:
        _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['solv_5ns_pair_dist'], dev_tol=tol)
        solv_5ns_viols.append(min(sum_viol) / len(sys['noe_index']['Dist_[A]']))
    except:
        solv_5ns_viols.append(err)

    try:
        _, sum_viol = summed_noe_violation(sys['noe_index']['Dist_[A]'], sys['ANI_min_pair_dist'], dev_tol=tol)
        ANI_min_viols.append(min(sum_viol)/len(sys['noe_index']['Dist_[A]']))
    except:
        ANI_min_viols.append(err)

x = np.arange(len(names))
width = 0.08  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 4*width, bl_viols, width, label='ETKDGv3')
rects2 = ax.bar(x - 3*width, noe_viols, width, label='NOE-ETKDG')

rects3 = ax.bar(x - 2*width, vac_min_viols, width, label='Vac Min')
rects4 = ax.bar(x - 1*width, solv_min_viols, width, label='Solv Min')
rects5 = ax.bar(x, ANI_min_viols, width, label='ANI Min')

rects6 = ax.bar(x + 1*width, vac_1ns_viols, width, label='Vac 1 ns')
rects7 = ax.bar(x + 2*width, solv_1ns_viols, width, label='Solv 1 ns')

rects8 = ax.bar(x + 3*width, vac_5ns_viols, width, label='Vac 5 ns')
rects9 = ax.bar(x  + 4*width, solv_5ns_viols, width, label='Solv 5 ns')



ax.set_ylabel('Best conf: sum(viol) per #restraints / A')
ax.set_title(f'NOE Restraint Violations (dev tol = {tol} A)')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.axhline(color='k')
plt.savefig(f'plots/noe/NOE_viol_tol_{tol}.png')
plt.show()


# NOE index plots
for i in range(len(best_noe)):
    x = np.arange(len(best_noe[i]))
    fig, ax = plt.subplots()
    plt.bar(x, best_noe[i])
    ax.set_xticks(x[::10])
    ax.set_xticklabels(x[::10], rotation=0)
    #plt.xticks(x, x)
    ax.set_xlabel(f'NOE index')
    ax.set_ylabel(f'NOE upper bound violation / A')
    ax.set_title(f'{names[i]}: Best NOE-ETKDG conformer NOE violations (tol = {tol} A)')
    plt.savefig(f'plots/noe/{names[i]}_tol_{tol}_noe_index.png')
    plt.show()


num = 12
x = np.arange(len(ten_best_noe[num][0]))
width = 0.16  # the width of the bars

fig, ax = plt.subplots(figsize=(16,6))
rects1 = ax.bar(x - 2*width, ten_best_noe[num][0], width, label='0')
rects2 = ax.bar(x - 1*width, ten_best_noe[num][1], width, label='1')

rects3 = ax.bar(x, ten_best_noe[num][2], width, label='2')

rects4 = ax.bar(x + 1*width, ten_best_noe[num][3], width, label='3')
rects5 = ax.bar(x + 2*width, ten_best_noe[num][4], width, label='4')

ax.set_ylabel('sum(viol) per #restraints / A')
ax.set_title(f'{names[num]}: NOE Restraint Violations (dev tol = {tol} A)')
#ax.set_xticks(x)
#ax.set_xticklabels(x[::10], rotation=0)
ax.set_xlabel(f'NOE index')
ax.set_ylabel(f'NOE upper bound violation / A')
ax.legend()
ax.axhline(color='k')
plt.savefig(f'plots/noe/{names[num]}NOE__noe_index.png')
plt.show()


print('d')
