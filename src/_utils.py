import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
import sys
import difflib
import os

def assign_hydrogen_pdbinfo(mol):
    """
    assumes all heavy atoms has complete PDB information
    """
    for idx,atm in enumerate(mol.GetAtoms()):
        if atm.GetPDBResidueInfo() is None and atm.GetAtomicNum() == 1:
            assert len(atm.GetNeighbors()) == 1, "String hydrogen with hypervalence at index {}".format(idx)
            heavy_atm = atm.GetNeighbors()[0]

            mi = Chem.AtomPDBResidueInfo()
            tmp = "H{}".format(heavy_atm.GetPDBResidueInfo().GetName().strip())
            mi.SetName("{: <4s}".format(tmp)) #the spacing is needed so that atom entries in the output pdb file can ALL be read
            mi.SetIsHeteroAtom(False)

            mi.SetResidueNumber(heavy_atm.GetPDBResidueInfo().GetResidueNumber())
            mi.SetResidueName(heavy_atm.GetPDBResidueInfo().GetResidueName())
            mi.SetChainId(heavy_atm.GetPDBResidueInfo().GetChainId())
            atm.SetMonomerInfo(mi)
    return mol
    

def map_mol_with_noe(mol, df, verbose = True):
    """
    The goal is to make the atom names in the df to be
    exactly the same as those in the mol naming

    mol: must already have named hydrogens added!!!
    """
    mol_atom_names2noe_atom_names = {}
    mol_atom_names2atom_index = {}
    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetAtomicNum() != 1: #currently only needs the hydrogen
            continue
        key = ("{}".format(atm.GetPDBResidueInfo().GetResidueNumber()) , "{}".format(atm.GetPDBResidueInfo().GetName().strip())) 
        mol_atom_names2noe_atom_names[key] = set()
        mol_atom_names2atom_index.setdefault(key, []).append(idx)


    noe_atom_pair2upper_distance = {}
    for _, row in df.iterrows():
        tup = ("{}".format(row["Residue_index_1"]) , "{}".format(row["Residue_name_1"]))
        #look for the atom name in the mol object that most resemble the given row's noe atom name
        key = difflib.get_close_matches(tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]], 1)[0]
        key = (tup[0], key) #add back the index
        mol_atom_names2noe_atom_names[key].add(tup)

        tmp = ("{}".format(row["Residue_index_2"]) , "{}".format(row["Residue_name_2"]))
        noe_atom_pair2upper_distance[(tup, tmp)] = row["Upper_bound_[nm]"] * 10 #XXX times 10 because of nm to A conversion
        noe_atom_pair2upper_distance[(tmp, tup)] = row["Upper_bound_[nm]"] * 10 
        #do again for the other atom of the atom pair
        tup = tmp
        key = difflib.get_close_matches(tup[1], [j[1] for j in mol_atom_names2noe_atom_names if j[0] == tup[0]], 1)[0]
        key = (tup[0], key) #add back the index
        mol_atom_names2noe_atom_names[key].add(tup)


    #XXX sanity check block
    noe_atom_names2mol_atom_names = {}
    for key,val in mol_atom_names2noe_atom_names.items():
        if len(val) == 1:
            if verbose:
                print("1-to-1 mapping: {}   {}".format(key, val))
            noe_atom_names2mol_atom_names[min(val)] = key #the min semantics is a 'hack' for obtaining the element in a set without popping it out

    if verbose:
        for key,val in mol_atom_names2noe_atom_names.items():
            if len(val) == 0:
                print("No NOE restraint for: {}   {}".format(key, val))

    #in this case usually go and change the NOE dataframe, 
    # because the atom names in the pdb file can actually have meaning when running MD
    trigger = False
    for key,val in mol_atom_names2noe_atom_names.items():
        if len(val) > 1:
            print("Error! non-unique mapping between: {}   {}".format(key, val))
            trigger = True

    if trigger: raise ValueError("Non Unique Mapping(s)")

    return mol_atom_names2atom_index, noe_atom_names2mol_atom_names, noe_atom_pair2upper_distance


def get_noe_restraint_bmat(mol, df):
    mol_atom_names2atom_index, noe_atom_names2mol_atom_names, noe_atom_pair2upper_distance = map_mol_with_noe(mol, df)

    bmat = AllChem.GetMoleculeBoundsMatrix(mol)
    print(mol_atom_names2atom_index)
    for key, val in noe_atom_pair2upper_distance.items():
        a = mol_atom_names2atom_index[noe_atom_names2mol_atom_names[key[0]]]
        b = mol_atom_names2atom_index[noe_atom_names2mol_atom_names[key[1]]]
        for item in product(a,b):
            if bmat[max(item), min(item)] > val: #updates the lower bound only if it is smaller than NOE suggests
                # print(bmat[max(item), min(item)] , val)
                bmat[max(item), min(item)] = val
            # print(item, bmat[min(item), max(item)], val)
            bmat[min(item), max(item)] = val
    
    return bmat

