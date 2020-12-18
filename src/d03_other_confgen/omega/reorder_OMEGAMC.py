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

from rdkit.RDLogger import logger
from rdkit.Chem import *

def AssignBondOrdersFromTemplate(refmol, mol):
    refmol2 = rdchem.Mol(refmol)
    mol2 = rdchem.Mol(mol)
    # do the molecules match already?
    matching = mol2.GetSubstructMatch(refmol2)
    if not matching:  # no, they don't match
        # check if bonds of mol are SINGLE
        for b in mol2.GetBonds():
            if b.GetBondType() != BondType.SINGLE:
                b.SetBondType(BondType.SINGLE)
                b.SetIsAromatic(False)
        # set the bonds of mol to SINGLE
        for b in refmol2.GetBonds():
            b.SetBondType(BondType.SINGLE)
            b.SetIsAromatic(False)
        # set atom charges to zero;
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)

        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        # do the molecules match now?
        if matching:
            if len(matching) > 1:
                print("More than one matching pattern found - picking one")
                #logger.warning("More than one matching pattern found - picking one")
            matching = matching[0]
            # apply matching: set bond properties
            for b in refmol.GetBonds():
                atom1 = matching[b.GetBeginAtomIdx()]
                atom2 = matching[b.GetEndAtomIdx()]
                b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
                b2.SetBondType(b.GetBondType())
                b2.SetIsAromatic(b.GetIsAromatic())
            # apply matching: set atom properties
            for a in refmol.GetAtoms():
                a2 = mol2.GetAtomWithIdx(matching[a.GetIdx()])
                a2.SetHybridization(a.GetHybridization())
                a2.SetIsAromatic(a.GetIsAromatic())
                a2.SetNumExplicitHs(a.GetNumExplicitHs())
                a2.SetFormalCharge(a.GetFormalCharge())
            #SanitizeMol(mol2) s
            if hasattr(mol2, '__sssAtoms'):
                mol2.__sssAtoms = None  # we don't want all bonds highlighted
        else:
            raise ValueError("No matching found")
    return mol2

pdb_code = '6fce'
os.chdir('../../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'
ref_pdb_path = pdb_code.lower() + '.pdb'
omega_pdb_path = f"30_{pdb_code.lower()}_OmegaMC_5400.pdb"

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')


# mdtraj
tmp = md.load(omega_pdb_path)
tmp.save("tmp.pdb")
tmp.save("mdtraj_" + pdb_code + '.pdb')

smiles_mol = Chem.MolFromSmiles(smiles)
#smiles_mol = Chem.AddHs(smiles_mol)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=False)
omega_mol = Chem.MolFromPDBFile("tmp.pdb", removeHs=False)

print(omega_mol.GetNumConformers())
print(omega_mol.GetNumAtoms())
print(ref_mol.GetNumAtoms())

# !!! Might require deactivating SanitizeMol() in AssignBondOrdersFromTemplate() !!!
ref_mol = AssignBondOrdersFromTemplate(smiles_mol, ref_mol)
omega_mol = AssignBondOrdersFromTemplate(smiles_mol, omega_mol)
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

Chem.MolToPDBFile(omega_mol, f'31_{pdb_code.lower()}_OmegaMC_restruct.pdb')



