from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
import numpy as np
import tqdm
import pandas as pd


smiles = "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C"
ref_pdb_path = "./CsA_atomname.pdb"

smiles_mol = Chem.MolFromSmiles(smiles)
#ref_mol = Chem.MolFromPDBFile("CsA_chcl3_norandcoord_5400.pdb", removeHs = False)
ref_mol = Chem.MolFromPDBFile("CsA_chcl3_norandcoord_5400.pdb", removeHs = True)
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)

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
            mi.SetName("{: <4s}".format(tmp))
            mi.SetResidueNumber(heavy_atm.GetPDBResidueInfo().GetResidueNumber())
            mi.SetResidueName(heavy_atm.GetPDBResidueInfo().GetResidueName())
            mi.SetChainId(heavy_atm.GetPDBResidueInfo().GetChainId())
            atm.SetMonomerInfo(mi)
    return mol


mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))



df = pd.read_csv("CsA_chcl3_noebounds.csv", sep = "\s", comment = "#")
out = []
for i in df["Residue_name_1"]:
    if i == "HN":
        out.append("H")
    elif i[1] != "C":
        out.append(i[0] + "C" + i[1:])
    else: out.append(i)
df["Residue_name_1"] = out

out = []
for i in df["Residue_name_2"]:
    if i == "HN":
        out.append("H")
    elif i[1] != "C":
        out.append(i[0] + "C" + i[1:])
    else: out.append(i)
df["Residue_name_2"] = out


named_index = np.array(["{}-{}".format(atm.GetPDBResidueInfo().GetResidueNumber(),atm.GetPDBResidueInfo().GetName().strip()) for atm in mol.GetAtoms()])

print(named_index)
print(mol.GetNumAtoms())
print(len(named_index))
print(df)

for idx,val in df.iterrows():
    a = np.nonzero(named_index == "{}-{}".format(val["Residue_index_1"], val["Residue_name_1"]))[0]
    b = np.nonzero(named_index == "{}-{}".format(val["Residue_index_2"], val["Residue_name_2"]))[0]
    print(a,b,val)
    for item in product(a,b):
        print(item)



conformers = np.array([Chem.Get3DDistanceMatrix(mol, i) for i in tqdm.tqdm(range(mol.GetNumConformers()))])
#conformers[:, 

