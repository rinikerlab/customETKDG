import tempfile
import pandas as pd
import sys
import pickle
import os
import glob
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import mdtraj as md

fname = "CsE_OMEGA_MC_numconf232.pdb"
#fname = "CsA_OMEGA_Macrocycle_numconf147.pdb"
path = "/home/kkajo/Workspace/Conformers/CsE/"
omega_pdb_path = path + fname
ref_pdb_path = "/home/kkajo/Workspace/Conformers/CsE/Ref/ref.pdb"
ref_pdb_path = "/home/kkajo/Workspace/Conformers/CsE/CsE_baseline_numconf5400.pdb"
smiles = "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C"

# mdtraj
tmp = md.load(omega_pdb_path)
tmp.save("tmp.pdb")
tmp.save(path + "mdtraj_" + fname)

smiles_mol = Chem.MolFromSmiles(smiles)
#smiles_mol = Chem.AddHs(smiles_mol)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=False)
omega_mol = Chem.MolFromPDBFile("tmp.pdb", removeHs=False)

print(omega_mol.GetNumConformers())
print(omega_mol.GetNumAtoms())
print(ref_mol.GetNumAtoms())

# !!! Might require deactivating SanitizeMol() in AssignBondOrdersFromTemplate() !!!
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, ref_mol)
omega_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, omega_mol)
order = list(omega_mol.GetSubstructMatches(ref_mol)[0])
omega_mol = Chem.RenumberAtoms(omega_mol, order)

#omega_mol = Chem.AddHs(omega_mol, addCoords=True)
#ref_mol_H = Chem.MolFromPDBFile(ref_pdb_path, removeHs=False)
#ref_mol_H = AllChem.AssignBondOrdersFromTemplate(smiles_mol, ref_mol_H)

info = []
for idx,atm in enumerate(ref_mol.GetAtoms()):
    if atm.GetAtomicNum() == 1:
        mi = atm.GetMonomerInfo()
        info.append(mi.GetName())
        #print(atm.GetPropNames())
        #print(dir(atm))

i = 0
for idx,atm in enumerate(omega_mol.GetAtoms()):
    if atm.GetAtomicNum() == 1:
        #print(idx)
        mi = atm.GetMonomerInfo()
        mi.SetName(info[i])
        atm.SetMonomerInfo(mi)
        i = i + 1

#omega_mol = Chem.AddHs(omega_mol, explicitOnly=False, addCords=True, boost=None, onlyOnAtoms=None, addResidueInfo=True)
#omega_mol = Chem.AddHs(omega_mol, addResidueInfo=True)



Chem.MolToPDBFile(omega_mol, "/home/kkajo/Workspace/Conformers/CsE/" + "Reordered_" + fname)

