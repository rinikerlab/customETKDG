from src.d06_analysis.analysis_utils import *
import os
import time
import pickle

t = time.time()

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
}
'''

systems = ['0CSA', '0CEC', '0CED', '1DP1', '1DP2', '1DP3', '1DP4', '1DP6', '2IFJ', '2L2X', '5LFF', '6BEU', '6BF3',
           '6FCE', '6HVB']
#systems = ['0CSA']

os.chdir('../../data/')

data = load_data(systems)

for sys in data:
    print(sys['name'])

    sys['bond_dist'] = get_noe_bond_dist(sys['noe_etkdg'][1], sys['noe_index'])

    sys['noe_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['noe_etkdg'][0], sys['ref'][0])
    sys['noe_tfd'] = get_tfd(sys['smiles'], sys['noe_etkdg'][0], sys['ref'][0])
    sys['noe_pair_dist'] = get_noe_pair_dist(sys['noe_etkdg'][1], sys['noe_index'])

    sys['bl_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['bl_etkdg'][0], sys['ref'][0])
    sys['bl_tfd'] = get_tfd(sys['smiles'], sys['bl_etkdg'][0], sys['ref'][0])
    sys['bl_pair_dist'] = get_noe_pair_dist(sys['bl_etkdg'][1], sys['noe_index'])

    try:
        sys['omega_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['omega'][0], sys['ref'][0])
        sys['omega_tfd'] = get_tfd(sys['smiles'], sys['omega'][0], sys['ref'][0])
        sys['omega_pair_dist'] = get_noe_pair_dist(sys['omega'][1], sys['noe_index'])
    except:
        sys['omega_rb_rmsd'], sys['omega_tfd'], sys['omega_pair_dist'] = None, None, None
        print('omega')

    try:
        sys['vac_min_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['vac_min'][0], sys['ref'][0])
        sys['vac_min_tfd'] = None  # get_tfd(sys['smiles'], sys['vac_min'][0], sys['ref'][0])
        sys['vac_min_pair_dist'] = get_noe_pair_dist(sys['vac_min'][1], sys['noe_index'])
    except:
        sys['vac_min_rb_rmsd'], sys['vac_min_tfd'], sys['vac_min_pair_dist'] = None, None, None
        print('vac min')

    try:
        sys['vac_1ns_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['vac_1ns'][0], sys['ref'][0])
        sys['vac_1ns_tfd'] = get_tfd(sys['smiles'], sys['vac_1ns'][0], sys['ref'][0])
        sys['vac_1ns_pair_dist'] = get_noe_pair_dist(sys['vac_1ns'][1], sys['noe_index'])
    except:
        sys['vac_1ns_rb_rmsd'], sys['vac_1ns_tfd'], sys['vac_1ns_pair_dist'] = None, None, None
        print('vac 1ns')

    try:
        sys['vac_5ns_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['vac_5ns'][0], sys['ref'][0])
        sys['vac_5ns_tfd'] = None  # get_tfd(sys['smiles'], sys['vac_5ns'][0], sys['ref'][0])
        sys['vac_5ns_pair_dist'] = get_noe_pair_dist(sys['vac_5ns'][1], sys['noe_index'])
    except:
        sys['vac_5ns_rb_rmsd'], sys['vac_5ns_tfd'], sys['vac_5ns_pair_dist'] = None, None, None
        print('vac 5ns')

    try:
        sys['solv_min_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['solv_min'][0], sys['ref'][0])
        sys['solv_min_tfd'] = get_tfd(sys['smiles'], sys['solv_min'][0], sys['ref'][0])
        sys['solv_min_pair_dist'] = get_noe_pair_dist(sys['solv_min'][1], sys['noe_index'])
    except:
        sys['solv_min_rb_rmsd'], sys['solv_min_tfd'], sys['solv_min_pair_dist'] = None, None, None
        print('solv min')

    try:
        sys['solv_1ns_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['solv_1ns'][0], sys['ref'][0])
        sys['solv_1ns_tfd'] = None  # get_tfd(sys['smiles'], sys['solv_1ns'][0], sys['ref'][0])
        sys['solv_1ns_pair_dist'] = get_noe_pair_dist(sys['solv_1ns'][1], sys['noe_index'])
    except:
        sys['solv_1ns_rb_rmsd'], sys['solv_1ns_tfd'], sys['solv_1ns_pair_dist'] = None, None, None
        print('solv 1ns')

    try:
        sys['solv_5ns_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['solv_5ns'][0], sys['ref'][0])
        sys['solv_5ns_tfd'] = None  # get_tfd(sys['smiles'], sys['solv_5ns'][0], sys['ref'][0])
        sys['solv_5ns_pair_dist'] = get_noe_pair_dist(sys['solv_5ns'][1], sys['noe_index'])
    except:
        sys['solv_5ns_rb_rmsd'], sys['solv_5ns_tfd'], sys['solv_5ns_pair_dist'] = None, None, None
        print('solv 5ns')

    try:
        sys['ANI_min_rb_rmsd'], _ = get_ring_beta_rmsd(sys['smiles'], sys['ANI_min'][0], sys['ref'][0])
        sys['ANI_min_tfd'] = get_tfd(sys['smiles'], sys['ANI_min'][0], sys['ref'][0])
        sys['ANI_min_pair_dist'] = get_noe_pair_dist(sys['ANI_min'][1], sys['noe_index'])
    except:
        sys['ANI_min_rb_rmsd'], sys['ANI_min_tfd'], sys['ANI_min_pair_dist'] = None, None, None
        print('ANI min')


pickle.dump(data, open('analysis_mintfd.data', 'wb'))

runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))