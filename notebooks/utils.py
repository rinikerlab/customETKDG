import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
import sys
import difflib
import os
import warnings

def assign_hydrogen_pdbinfo(mol):
    """
    assumes all heavy atoms have complete PDB information
    """
    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetPDBResidueInfo() is None and atm.GetAtomicNum() == 1:
            assert len(atm.GetNeighbors()) == 1, "String hydrogen with hypervalence at index {}".format(idx)
            heavy_atm = atm.GetNeighbors()[0]

            mi = Chem.AtomPDBResidueInfo()
            tmp = "H{}".format(heavy_atm.GetPDBResidueInfo().GetName().strip())
            mi.SetName("{: <4s}".format(
                tmp))  # the spacing is needed so that atom entries in the output pdb file can ALL be read
            mi.SetIsHeteroAtom(False)

            mi.SetResidueNumber(heavy_atm.GetPDBResidueInfo().GetResidueNumber())
            mi.SetResidueName(heavy_atm.GetPDBResidueInfo().GetResidueName())
            mi.SetChainId(heavy_atm.GetPDBResidueInfo().GetChainId())
            atm.SetMonomerInfo(mi)
    return mol


def map_mol_with_noe(mol, df, verbose=True):
    """
    The goal is to make the atom names in the df to be
    exactly the same as those in the mol naming

    mol: must already have named hydrogens added!!!
    """
    mol_atom_names2noe_atom_names = {}
    mol_atom_names2atom_index = {}
    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetAtomicNum() != 1:  # currently only needs the hydrogen
            continue
        key = (
        "{}".format(atm.GetPDBResidueInfo().GetResidueNumber()), "{}".format(atm.GetPDBResidueInfo().GetName().strip()))
        mol_atom_names2noe_atom_names[key] = set()
        mol_atom_names2atom_index.setdefault(key, []).append(idx)

    # divide @@ cases into two methyl groups
    for idx, row in df.iterrows():
        if row["Residue_name_1"] in "HD@@":
            add_row = row.copy()
            row["Residue_name_1"] = "HD1"
            add_row["Residue_name_1"] = "HD2"
            df.drop(idx, inplace=True)
            df = df.append(row)
            df = df.append(add_row)
    for idx, row in df.iterrows():
        if row["Residue_name_2"] in "HD@@":
            add_row = row.copy()
            row["Residue_name_2"] = "HD1"
            add_row["Residue_name_2"] = "HD2"
            df.drop(idx, inplace=True)
            df = df.append(row)
            df = df.append(add_row)

    noe_atom_pair2upper_distance = {}
    for _, row in df.iterrows():
        tup = ("{}".format(row["Residue_index_1"]), "{}".format(row["Residue_name_1"]))
        if tup[1] in ["HB1", "HB2", "HB@"]:
            tup = (tup[0], "HB")

        # look for the atom name in the mol object that most resemble the given row's noe atom name
        try:
            key = \
                difflib.get_close_matches(tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]], 1,
                                          0.2)[0]
        except:
            raise ValueError("Residue {}: Could not find match between NOE atom {} and PDB atoms {}."
                             .format(tup[0], tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]]))

        key = (tup[0], key)  # add back the index
        mol_atom_names2noe_atom_names[key].add(tup)

        tmp = ("{}".format(row["Residue_index_2"]), "{}".format(row["Residue_name_2"]))
        if tmp[1] in ["HB1", "HB2", "HB@"]:
            tmp = (tmp[0], "HB")

        noe_atom_pair2upper_distance[(tup, tmp)] = row["Upper_bound_[nm]"] * 10 # times  10 because of nm to A conv
        noe_atom_pair2upper_distance[(tmp, tup)] = row["Upper_bound_[nm]"] * 10

        # do again for the other atom of the atom pair
        tup = tmp
        try:
            key = difflib.get_close_matches(tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]], 1,
                                            0.2)[0]
        except:
            raise ValueError("Residue {}: Could not find match between NOE atom {} and PDB atoms {}."
                             .format(tup[0], tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]]))

        key = (tup[0], key)  # add back the index
        mol_atom_names2noe_atom_names[key].add(tup)

    # XXX sanity check block
    noe_atom_names2mol_atom_names = {}
    for key, val in mol_atom_names2noe_atom_names.items():
        if len(val) == 1:
            if verbose:
                print("1-to-1 mapping: {}   {}".format(key, val))
            noe_atom_names2mol_atom_names[min(
                val)] = key  # the min semantics is a 'hack' for obtaining the element in a set without popping it out

    if verbose:
        for key, val in mol_atom_names2noe_atom_names.items():
            if len(val) == 0:
                print("No NOE restraint for: {}   {}".format(key, val))

    # in this case usually go and change the NOE dataframe,
    # because the atom names in the pdb file can actually have meaning when running MD
    trigger = False
    for key, val in mol_atom_names2noe_atom_names.items():
        if len(val) > 1:
            warnings.warn("Non-unique mapping between PDB atom {} and NOE atoms {}.".format(key, val))
            pick = query_yes_no("Pick most probable NOE atom {} (yes) or exit (no)?"
                         .format(difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]))
            if pick:
                val = difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]
                val = {(key[0], val)}  # add back the index
                # mol_atom_names2noe_atom_names.pop(key, None) # remove non-unique
                mol_atom_names2noe_atom_names[key] = val  # re-add most probable
                print("Chosen mapping: {}   {}".format(key, val))
                noe_atom_names2mol_atom_names[min(val)] = key
            else:
                raise ValueError("Non Unique Mapping(s)")

    if trigger: raise ValueError("Non Unique Mapping(s)")

    return mol_atom_names2atom_index, noe_atom_names2mol_atom_names, noe_atom_pair2upper_distance


def get_noe_restraint_bmat(mol, df):
    tot, err = 0, 0
    mol_atom_names2atom_index, noe_atom_names2mol_atom_names, noe_atom_pair2upper_distance = map_mol_with_noe(mol, df)
    bmat = AllChem.GetMoleculeBoundsMatrix(mol)
    #print("mol_atom_names2atom_index:")
    #print(mol_atom_names2atom_index)
    for key, val in noe_atom_pair2upper_distance.items():
        tot = tot + 1
        try:
            a = mol_atom_names2atom_index[noe_atom_names2mol_atom_names[key[0]]]
            b = mol_atom_names2atom_index[noe_atom_names2mol_atom_names[key[1]]]
            for item in product(a, b):
                if bmat[max(item), min(item)] > val:  # updates the lower bound only if it is smaller than NOE suggests
                    # print(bmat[max(item), min(item)] , val)
                    bmat[max(item), min(item)] = val
                # print(item, bmat[min(item), max(item)], val)
                bmat[min(item), max(item)] = val  # upper bound?
        except:
            warnings.warn("NOE atom {} or {} not found in PDB atoms. Not setting restraint.".format(key[0], key[1]))
            err = err + 1
    print("Set {} NOE restraints, not able to assign {} restraints.".format(tot-err, err))
    return bmat


def query_yes_no(question, default="yes"):
    """
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")