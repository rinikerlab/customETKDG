from src.d05_refinement.refinement_utils import minimise_energy_confs
import time
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

t = time.time()

# change these
simtime_ns = 0.1
solventDielectric = 78
confs_index = list(range(0, 1))  # if you do not want to simulate all conformers, needed if all_confs=False
pdb_code = '6beu'

pdb_path = f'21_{pdb_code}_tol1_noebounds_5400.pdb'
rdmol_df_path = f'10_{pdb_code}_NOE_index.csv'
os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'

reporter_path = os.getcwd()+'/'+str(simtime_ns)+'ns_solv_reporters/'
if not os.path.exists(reporter_path):
    os.makedirs(reporter_path)

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')

smiles_mol = Chem.MolFromSmiles(smiles)
rdmol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=False)
rdmol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, rdmol)
AllChem.AlignMolConformers(rdmol)
print(rdmol.GetNumAtoms())
rdmol_df = pd.read_csv(rdmol_df_path, index_col=0)
#Chem.AssignStereochemistryFrom3D(rdmol, confId=0)


out_mol = minimise_energy_confs(rdmol, noe_restraints=rdmol_df, allow_undefined_stereo=True, restraints=True,
                                use_mlddec=True, implicitSolvent=True, solventDielectric=solventDielectric,
                                simtime_ns=simtime_ns, hdf5_reporter_path=reporter_path, all_confs=False,
                                confs_index=confs_index)

for i in range(rdmol.GetNumConformers()):
    AllChem.AlignMol(prbMol=out_mol, refMol=rdmol, prbCid=i, refCid=i)

Chem.MolToPDBFile(out_mol, f"51_{simtime_ns}ns_vac_{min(confs_index)}-{max(confs_index)}idx_{pdb_path}")

runtime = time.time() - t
print(f"Took {runtime:.0f} s to minimize {len(confs_index)} conformers.")