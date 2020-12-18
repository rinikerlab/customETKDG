import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import TorsionFingerprints
from rdkit.Chem import AllChem
import mdtraj as md
import cpeptools
import tempfile
import tqdm
import matplotlib.pyplot as plt

def load_data(systems):
    """

    :type pdb_list: object
    """
    data = []
    for sys in systems:
        #info = [sys, get_solvent(sys), get_dielectric(sys)]
        noe_index = pd.read_csv(sys + '/' + f'10_{sys.lower()}_NOE_index.csv', index_col=0)

        smiles_path = sys + '/' + sys.lower() + '.smi'
        with open(smiles_path, "r") as tmp:
            smiles = tmp.read().replace("\n", "")
            smiles = smiles.replace('N+H2', 'NH2+')
            smiles = smiles.replace('N+H3', 'NH3+')

        ref_pdb_path = sys + '/' + sys.lower() + '.pdb'
        ref = [ref_pdb_path, Chem.MolFromPDBFile(ref_pdb_path, removeHs = False)]

        bl_path = sys + '/' + f'20_{sys.lower()}_baseline_5400.pdb'
        bl = [bl_path, Chem.MolFromPDBFile(bl_path, removeHs = False)]

        noe_etkdg_path = sys + '/' + f'21_{sys.lower()}_tol1_noebounds_5400.pdb'
        noe_etkdg = [noe_etkdg_path, Chem.MolFromPDBFile(noe_etkdg_path, removeHs = False)]

        try:
            omega_path = sys + '/' + f'31_{sys.lower()}_OmegaMC_restruct.pdb'
            omega = [omega_path, Chem.MolFromPDBFile(omega_path, removeHs = False)]
        except:
            omega = [None, None]
            print(f'No OMEGA file for {sys}.')

        vac_min_path = sys + '/' + f'40_min_vac_{sys.lower()}_tol1_noebounds_5400.pdb'
        vac_min = [vac_min_path, Chem.MolFromPDBFile(vac_min_path, removeHs = False)]

        vac_1ns_path = sys + '/' + f'41_1ns_vac_{sys.lower()}_tol1_noebounds_540.pdb'
        vac_1ns = [vac_1ns_path, Chem.MolFromPDBFile(vac_1ns_path, removeHs = False)]

        try:
            vac_5ns_path = sys + '/' + f'42_5ns_vac_{sys.lower()}_tol1_noebounds_54.pdb'
            vac_5ns = [vac_5ns_path, Chem.MolFromPDBFile(vac_5ns_path, removeHs = False)]
        except:
            vac_5ns = [None, None]
            print(f'No vac_5ns files for {sys}.')

        solv_min_path = sys + '/' + f'50_min_solv_{sys.lower()}_tol1_noebounds_5400.pdb'
        solv_min = [solv_min_path, Chem.MolFromPDBFile(solv_min_path, removeHs = False)]

        try:
            solv_1ns_path = sys + '/' + f'51_1ns_solv_{sys.lower()}_tol1_noebounds_540.pdb'
            solv_1ns = [solv_1ns_path, Chem.MolFromPDBFile(solv_1ns_path, removeHs = False)]
        except:
            solv_1ns = [None, None]
            print(f'No solv_1ns files for {sys}.')

        try:
            solv_5ns_path = sys + '/' + f'52_5ns_solv_{sys.lower()}_tol1_noebounds_54.pdb'
            solv_5ns = [solv_5ns_path, Chem.MolFromPDBFile(solv_5ns_path, removeHs = False)]
        except:
            solv_5ns = [None, None]
            print(f'No solv_5ns files for {sys}.')

        try:
            ANI_min_path = sys + '/' + f'60_min_ANI_{sys.lower()}_tol1_noebounds_5400.pdb'
            ANI_min = [ANI_min_path, Chem.MolFromPDBFile(ANI_min_path, removeHs = False)]
        except:
            ANI_min = [None, None]
            print(f'No ANI_min files for {sys}.')

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
            'solv_1ns': solv_1ns,
            'solv_5ns': solv_5ns,
            'ANI_min': ANI_min
        }
        data.append(dict)
        # [info, noe_index, ref, bl, noe_etkdg, omega, vac_min, vac_1ns, vac_5ns, solv_min, ANI_min]
    return data


def get_solvent(name):
    names = ['0CSA', '0CEC', '0CED', '1DP1', '1DP2', '1DP3', '1DP4', '1DP5', '1DP6', '2FRB', '2IFJ', '2L2X', '2MOA',
             '2MUH', '2N7N', '2NBC', '5LFF', '6B34', '6BEU', '6BF3', '6FCE', '6HVB', '6HVC', '6VY8']
    solvent = ['CHCl3', 'CCl4', 'DMSO', 'CHCl3', 'CHCl3', 'CHCl3', 'CHCl3', 'CHCl3', 'CHCl3', 'NONE', 'D2O/H2O',
               'chloroform-d/ethanol-d5 5:1', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'H20/MeCN 1:1',
               'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O', 'D2O/H2O']
    assert len(names) == len(solvent)
    return solvent[names.index(name)]


def get_dielectric(name):
    names = ['0CSA', '0CEC', '0CED', '1DP1', '1DP2', '1DP3', '1DP4', '1DP5', '1DP6', '2FRB', '2IFJ', '2L2X', '2MOA',
             '2MUH', '2N7N', '2NBC', '5LFF', '6B34', '6BEU', '6BF3', '6FCE', '6HVB', '6HVC', '6VY8']
    dielectric = [4.8, 2.2, 48, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, float("NaN"), 78, 10, 78, 78, 78, 78, 78, 56, 78, 78, 78,
                  78, 78, 78]
    assert len(names) == len(dielectric)
    return dielectric[names.index(name)]


def get_ring_beta_rmsd(smiles, pdb_path, ref_pdb_path): #not full rmsd, just ring + beta atom rmsd
    smiles_mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
    mol = Chem.MolFromPDBFile(pdb_path)

    ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)
    mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol)
    order = list(mol.GetSubstructMatches(ref_mol)[0])
    mol = Chem.RenumberAtoms(mol, order)

    indices = cpeptools.get_largest_ring(ref_mol)
    indices = cpeptools.mol_ops.get_neighbour_indices(ref_mol, indices)
    #assert len(set(indices) - set(cpeptools.get_largest_ring(mol))) == 0, "ring atom indices do not agree"

    tmp_dir = tempfile.mkdtemp()
    ref_pdb_filename = tempfile.mktemp(suffix=".pdb", dir = tmp_dir)
    pdb_filename = tempfile.mktemp(suffix=".pdb", dir = tmp_dir)
    # chem add Hs
    Chem.MolToPDBFile(ref_mol, ref_pdb_filename)
    Chem.MolToPDBFile(mol, pdb_filename)

    ref  = md.load(ref_pdb_filename)
    #ref = ref.center_coordinates()
    compare = md.load(pdb_filename)
    #compare = compare.center_coordinates()
    #print(compare, mol.GetNumConformers())
    #print(" {} has {} conformers".format(smiles, len(compare)))

    bb_rmsd = md.rmsd(compare, ref, 0, atom_indices = indices)
    compare = compare.superpose(ref, 0, atom_indices = indices)
    return bb_rmsd, compare[np.argmin(bb_rmsd)]


def get_tfd(smiles, pdb_path, ref_pdb_path):
    smiles_mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
    mol = Chem.MolFromPDBFile(pdb_path)

    ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)
    mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol)
    order = list(mol.GetSubstructMatches(ref_mol)[0])
    mol = Chem.RenumberAtoms(mol, order)

    tfd = np.array([Chem.TorsionFingerprints.GetTFDBetweenMolecules(mol, ref_mol, confId1=i, confId2=0) for i in tqdm.tqdm(range(mol.GetNumConformers()))])
    return tfd


def get_noe_pair_dist(rdmol, noe_index):
    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(rdmol, i) for i in tqdm.tqdm(range(rdmol.GetNumConformers()))])
    distances = distance_matrix_for_each_conformer[:, noe_index['Atm_idx_1'], noe_index['Atm_idx_2']]
    return distances


def get_noe_bond_dist(rdmol, noe_index):
    bond_distance_matrix = Chem.GetDistanceMatrix(rdmol, 0)
    distances = bond_distance_matrix[noe_index['Atm_idx_1'], noe_index['Atm_idx_2']]
    return distances


def summed_noe_violation(noe_pair_dist, meth_pair_dist, dev_tol=0):
    violations = []
    for i in range(len(meth_pair_dist)):
        tmp = (meth_pair_dist[i] - noe_pair_dist) - dev_tol
        tmp[tmp < 0] = 0
        violations.append(tmp)

    sum_viol = []
    for i in range(len(violations)):
        sum_viol.append(np.sum(violations[i]))

    return violations, sum_viol

def set_violin_color(vp, markercolor, bodycolor):
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vps = vp[partname]
        vps.set_edgecolor(markercolor)
        vps.set_linewidth(1)
    for pc in vp['bodies']:
        pc.set_facecolor(bodycolor)
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)


def make_violin_plot_2(noe_index, bond_dist, pair_dist_1, pair_dist_2, plotname):
    widths = 0.3
    offset = 0.3
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 0.6*len(bond_dist)))

    # ax1.set_title('Basic Plot')
    red_square = dict(markerfacecolor='r', marker='o')
    blue_square = dict(markerfacecolor='b', marker='o')

    plot_1 = pair_dist_1 - np.array(noe_index['Dist_[A]'])
    plot_2 = pair_dist_2 - np.array(noe_index['Dist_[A]'])
    labels = bond_dist.astype(int)
    #for i in range(len(labels)):
    #    labels, plot_1[i,:], plot_2[i,:] = (list(t) for t in zip(*sorted(zip(labels, plot_1[i,:], plot_2[i,:]))))

    vp_1 = ax1.violinplot([plot_1[:, i] for i in range(plot_1.shape[1])], vert=False, widths=widths,
                          positions=np.array(range(plot_1.shape[1])), showmeans=False, showmedians=True,
                          showextrema=True)
    vp_2 = ax1.violinplot([plot_2[:, i] for i in range(plot_2.shape[1])], vert=False, widths=widths,
                          positions=np.array(range(plot_2.shape[1])), showmeans=False, showmedians=True,
                          showextrema=True)

    set_violin_color(vp_1, "red", "lightcoral")
    set_violin_color(vp_2, "blue", "lightblue")

    ax1.set_xlabel('Distance - Upper NOE Restraint / Å')
    plt.grid(axis="x", which="both")
    plt.yticks(range(plot_1.shape[1]), labels)
    plt.ylim(0 - 0.5, plot_1.shape[1] - 0.5)
    #plt.xlim(-5, 20)
    plt.axvline(x=0)
    plt.legend()
    plt.savefig(f'plots/{plotname}_distplot.png')
    return 0
