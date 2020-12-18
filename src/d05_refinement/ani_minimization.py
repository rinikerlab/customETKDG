from rdkit.Geometry import Point3D
from ase.optimize import BFGS
from ase import Atoms
import torchani
import os, re
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from src.d05_refinement.refinement_utils import tqdm_joblib


def min_ani(molecular_formula, mol, i):
    atoms = Atoms(molecular_formula)
    atoms.positions = mol.GetConformer(i).GetPositions()
    calculator = torchani.models.ANI2x().ase()  # ANI1cc
    atoms.set_calculator(calculator)

    dyn = BFGS(atoms, logfile='ANI_log.log')
    # https://wiki.fysik.dtu.dk/ase/ase/optimize.html?highlight=bfgs
    dyn.run(fmax=0.05, steps=10*1000)  # 0.05 what unit is this?

    coords = []
    for j in range(mol.GetNumAtoms()):
        coords.append(atoms.positions[j])
    return coords


t = time.time()

pdb_code = '0ced'

pdb_path = f'21_{pdb_code}_tol1_noebounds_5400.pdb'
os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')


smiles_mol = Chem.MolFromSmiles(smiles)
mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=False)
mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, mol)

molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

symbol_order = re.split("[0-9]+", molecular_formula)[:-1]

atom_order = []
for s in symbol_order:
    for atom in mol.GetAtoms():
        if s != atom.GetSymbol(): continue
        atom_order.append(atom.GetIdx())

back_order = []
for a in range(len(atom_order)):
    back_order.append(atom_order.index(a))

out_mol = copy.deepcopy(mol)
out_mol = Chem.RenumberAtoms(out_mol, atom_order)

numConf = out_mol.GetNumConformers()

# parallelize expensive simulations
with tqdm_joblib(tqdm(desc="ANI-minimization of conformers", total=numConf)) as progress_bar:
    coords_set = Parallel(n_jobs=-1)(delayed(min_ani)(molecular_formula, out_mol, i) for i in range(numConf))

# assign new coords
for i in range(numConf):
    for j in range(out_mol.GetNumAtoms()):
        coords = coords_set[i]
        out_mol.GetConformer(i).SetAtomPosition(j, Point3D(*coords[j]))

# sort back
out_mol = Chem.RenumberAtoms(out_mol, back_order)
# align
for i in range(mol.GetNumConformers()):
    AllChem.AlignMol(prbMol=out_mol, refMol=mol, prbCid=i, refCid=i)

Chem.MolToPDBFile(out_mol, f"60_min_ANI_{pdb_path}")

runtime = time.time() - t
print(f"Took {runtime:.0f} s to minimize {out_mol.GetNumConformers()} conformers.")