import tempfile
import mdtraj as md
from .mol_ops import *
from rdkit import Chem
from rdkit.Chem import AllChem
import tqdm

def get_ring_rmsd(mol, ref_pdb_path, ref_compare = "all", beta_atoms = True): 

    smiles_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path)

    ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)
    order = list(ref_mol.GetSubstructMatches(Chem.RemoveHs(mol))[0])
    ref_mol = Chem.RenumberAtoms(ref_mol, order)

    selection = ref_compare
    if selection == "all":
        selection = set(range(mol.GetNumConformers() + 1))
    elif isinstance(selection, Iterable) and type(selection) is not str:
        selection = [int(i) for i in selection] #need to perserve order so cannot use set
    else:
        raise ValueError("Unrecognized ref_compare type {}".format(type(resid)))

    indices = get_largest_ring(ref_mol)
    if beta_atoms:
        indices = get_neighbour_indices(ref_mol, indices)
    #assert len(set(indices) - set(cpeptools.get_largest_ring(mol))) == 0, "ring atom indices do not agree"

    #FIXME refactor
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

    # compare = compare.superpose(ref, 0, atom_indices = indices)
    # return bb_rmsd, compare[np.argmin(bb_rmsd)]

    bb_rmsd = np.array([md.rmsd(compare, ref, i, atom_indices = indices) for i in selection]) * 10 #convert to angstrom
    return bb_rmsd

def get_torsion_rmsd(mol, ref_pdb_path, ref_compare = "all"):
    smiles_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path)

    ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)
    order = list(ref_mol.GetSubstructMatches(Chem.RemoveHs(mol))[0])
    ref_mol = Chem.RenumberAtoms(ref_mol, order)

    selection = ref_compare
    if selection == "all":
        selection = set(range(mol.GetNumConformers() + 1))
    elif isinstance(selection, Iterable) and type(selection) is not str:
        selection = [int(i) for i in selection] #need to perserve order so cannot use set
    else:
        raise ValueError("Unrecognized ref_compare type {}".format(type(resid)))

    indices = get_largest_ring(ref_mol)
    indices_tor = [indices[i : i + 4] for i in range(len(indices) - 3)]

    #FIXME refactor, with tqdm
    tmp_dir = tempfile.mkdtemp()
    ref_pdb_filename = tempfile.mktemp(suffix=".pdb", dir = tmp_dir)
    pdb_filename = tempfile.mktemp(suffix=".pdb", dir = tmp_dir)
    Chem.MolToPDBFile(ref_mol, ref_pdb_filename)
    Chem.MolToPDBFile(mol, pdb_filename)

    ref  = md.load(ref_pdb_filename)
    compare = md.load(pdb_filename)
    # print(" {} has {} conformers".format(smiles, len(compare)))

    out = []
    ref_angles_arr = md.compute_dihedrals(ref, indices_tor)
    for ref_angles in ref_angles_arr:
        compare_angles = md.compute_dihedrals(compare, indices_tor)

        tmp = compare_angles - ref_angles
        tmp = (tmp + np.pi) % (2 * np.pi) - np.pi
        
        # tmp = np.absolute(tmp)
        # tmp = np.mean(tmp, axis = 1)
        tmp = np.sqrt(np.mean(tmp**2, axis = 1))

        rmsd_angles_array = tmp * 180 / np.pi
        out.append(rmsd_angles_array)

    out = np.array(out)
    return out
    # compare = compare.superpose(ref, 0, atom_indices = indices)
    # return mean_angles_array, compare[np.argmin(mean_angles_array)]



def get_noe_pair_dist(mol, df = None): #TODO what if you want to compare against minimised structure? overwrite whenever a conformer is minimised?
    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(mol, i) for i in tqdm.tqdm(range(mol.GetNumConformers()))])

    if hasattr(mol, "distance_upper_bounds"): 
        df = mol.distance_upper_bounds #FIXME
    distances = distance_matrix_for_each_conformer[:, df.idx1, df.idx2] - np.array(df.distance)
    #- df.distance #XXX more efficient calculate with only the needed pairs
    sum_violations = np.copy(distances)
    sum_violations[sum_violations < 0] = 0
    return distances, np.sum(sum_violations, axis = 1)


def weighted_upper_violation(mol, which_confs, weights = None, df = None):
    which_confs = [int(i) for i in which_confs]
    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(mol, i) for i in tqdm.tqdm(which_confs)])

    if hasattr(mol, "distance_upper_bounds"): 
        df = mol.distance_upper_bounds #FIXME

    distances = distance_matrix_for_each_conformer[:, df.idx1, df.idx2] - np.array(df.distance)
    # assert len(weights) == len(distances), "dimension mismatch"
    weighted_avg_distance = np.average(distances, axis = 0, weights = weights)
    #- df.distance #XXX more efficient calculate with only the needed pairs
    sum_violations = np.copy(weighted_avg_distance)
    sum_violations[sum_violations < 0] = 0

    return sum(sum_violations)

def _weighted_noe_violation_helper(mol, which_confs, noe, weights = None):

    """account for chemcial equivalence

    not only can look at summed violation, but any other
    """
    which_confs = [int(i) for i in which_confs]
    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(mol, i) for i in which_confs])

    df = noe.add_noe_to_mol(mol, remember_chemical_equivalence = True).distance_upper_bounds
    distances = distance_matrix_for_each_conformer[:, df.idx1, df.idx2] - np.array(df.distance)

    distances_by_chemical_equivalence = np.split(distances, np.unique(noe.chemical_equivalence_list, return_index=True)[1][1:], axis = 1) #here I group the distances by their chemical equivalence track

    distances = np.stack([np.mean(d, axis = 1) for d in distances_by_chemical_equivalence], axis = 1)
    # print(distances[0].shape, len(distances))

    weighted_avg_distance = np.average(distances, axis = 0, weights = weights) #weighted over the conformer bundle
    sum_violations = np.copy(weighted_avg_distance)

    return sum_violations

def weighted_noe_violation(mol, which_confs, noe, weights = None, agg_func = np.sum):
    sum_violations = _weighted_noe_violation_helper(mol, which_confs, noe, weights)
    return agg_func(sum_violations)


def weighted_noe_upper_violation(mol, which_confs, noe, weights = None, agg_func = np.sum):

    sum_violations = _weighted_noe_violation_helper(mol, which_confs, noe, weights)
    sum_violations[sum_violations < 0] = 0

    return agg_func(sum_violations)


#TODO below, new file for plots?
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

