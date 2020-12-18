from rdkit import Chem
import tqdm
import numpy as np
from itertools import product
from src.d00_utils.utils import assign_hydrogen_pdbinfo, map_mol_with_noe

# No longer needed function. We now directly generate the consistent NOE_index file with the conformers.
def get_distances(mol, noe_df):
    mol_atom_names2atom_index, noe_atom_names2mol_atom_names, noe_atom_pair2upper_distance = map_mol_with_noe(mol,
                                                                                                              noe_df,
                                                                                                              verbose=False)

    index_1, index_2 = [], []  # symmetric, thus two lists are interchangable
    labels = []
    ref_val = []
    res_dist = []  # distance between residuals
    bond_dist = []  # number of bonds between atoms

    bond_order = Chem.rdmolops.GetDistanceMatrix(mol)

    prev_a = 0
    prev_b = 0

    for key, val in noe_atom_pair2upper_distance.items():
        tmp_1 = noe_atom_names2mol_atom_names[key[0]]
        tmp_2 = noe_atom_names2mol_atom_names[key[1]]
        # print(key[0])
        # print(tmp_1)
        a = mol_atom_names2atom_index[tmp_1]
        # print(tmp_2[0])
        b = mol_atom_names2atom_index[tmp_2]

        if not (b == prev_a and a == prev_b):  # ignore reversed pair
            for item in product(a, b):
                # print(item)
                index_1.append(item[0])  # RDKit indexing starts at 0!
                index_2.append(item[1])
                ref_val.append(val)

                r1 = int(tmp_1[0])  # string to int conversion
                r2 = int(tmp_2[0])
                if abs(r1 - r2) <= 5:  # take the shorter route along the ring, 11 residuals total
                    dist = abs(r1 - r2)
                else:
                    dist = abs(abs(r1 - r2) - 11)
                res_dist.append(dist)

                bd = int(bond_order[item[0], item[1]])
                bond_dist.append(bd)

                labels.append(
                    "{}-{} {}R {}B {}:{}{}-{}:{}{} ({:0.2f})".format(tmp_1[0], tmp_2[0], dist, bd, item[0], tmp_1[0],
                                                                     tmp_1[1], item[1], tmp_2[0], tmp_2[1], val))

        prev_a = a
        prev_b = b

    # sort everything for bond_dist (first arg in zip)
    res_dist, bond_dist, labels, ref_val, index_1, index_2 = (list(t) for t in zip(
        *sorted(zip(bond_dist, res_dist, labels, ref_val, index_1, index_2))))

    distance_matrix_for_each_conformer = np.array(
        [Chem.Get3DDistanceMatrix(mol, i) for i in tqdm.tqdm(range(mol.GetNumConformers()))])
    dist = distance_matrix_for_each_conformer[:, index_1, index_2]

    ref_val = np.transpose(np.array(ref_val)) # convert list to array
    dist = np.transpose(dist)

    return ref_val, dist


def get_distances_new(rdmol, noe_index):
    noe_dist = 0
    confs_dist = 0

    distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(rdmol, i) for i in tqdm.tqdm(range(rdmol.GetNumConformers()))])
    distances = distance_matrix_for_each_conformer[:, noe_index['Atm_idx_1'], noe_index['Atm_idx_2']]

    return noe_index['Dist_[A]'], distances