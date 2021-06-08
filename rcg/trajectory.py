import tempfile
import mdtraj as md
import numpy as np
from rdkit import Chem

class Trajectory():
    """
    should store the seed conformer, the NOE data and topology
    - simulation frames (this can be big in size! how to handle)
    - which conf as seed 
    - dielectric  
    - restraints (keep the openmm force?) + length of simulation and basic info (openmm system?)
    - what about torchani?

    frame loaded in as mdtraj _trajectory

    probably does not need to store it as mdtraj, just parmed and coordinates
    """

    def __init__(self):
        self.frame_separation = None
        self._trajectory = None # mdtraj object
    
    def from_parmed(self, structure, smiles = None):
        self.structure = structure
        if self.structure.title: #XXX more valid check to see if the supplied smiles actually matches the molecule?
            self.smiles = self.structure.title
        # else:
        #     if smiles is None:
        #         raise ValueError("Require valid SMILES of the molecule.")
        #     else:
        #         self.smiles = smiles

    def copy_names_from_mol(self, mol, resnum = [0]): 
        orig_top = self._trajectory.topology

        tmp_dir = tempfile.mkdtemp()
        file_name = "{}/mol.pdb".format(tmp_dir)
        Chem.MolToPDBFile(mol, file_name, 0)
        new_top = md.load(file_name).topology #only has 'solute' mol

        indices = orig_top.select(" or ".join(["(resid == {} )".format(i) for i in resnum]))

        for new_idx, orig_idx in enumerate(indices):
            assert new_top.atom(new_idx).element.number == orig_top.atom(orig_idx).element.number, ValueError("Topology mismatch.")
        
        chain = new_top.chain(0) #XXX assume system is simple enough that only has one chain
        for res in orig_top.residues:
            if res.index not in resnum:
                new_res = new_top.add_residue(res.name, chain)
                for atom in res.atoms:
                    new_top.add_atom(atom.name, atom.element, new_res, serial = atom.serial)

        # XXX the following registers the correct bond connectivity,
        # but due to atom name changes, if I write out of the whole system
        # e.g. including solvents it has problems, but with the kind of normal
        # systems we have, the correct bond connectivity can be inferred
        
        # for bond in orig_top.bonds:
        #     a1, a2 = bond
        #     new_top.add_bond(a1, a2, type=bond.type, order=bond.order)

        self._trajectory.topology = new_top   

    def from_mol(self, rdmol):
        self.structure = parmed.rdkit.load_rdkit(rdmol)

    def add_frames(self, xyz, unit = "nanometers"):
        if self._trajectory is None: #no frames yet
            xyz = np.array(xyz)
            self._trajectory = md.Trajectory(xyz, md.Topology().from_openmm(self.structure.topology))
        else:
            assert self._trajectory.xyz.shape == xyz.shape, ValueError("Input frame shape {} does not match existing trajectory frame shape {}.".format(xyz.shape, self._trajectory.xyz.shape))
            self._trajectory.xyz = np. concatenate([self._trajectory.xyz, xyz])

    def to_mol(self):
        """
        trajectory frames into conformers
        """
        raise NotImplementedError
    
    def to_mdtraj(self):
        return self._trajectory

    def to_xyz(self):
        """ in angstroms
        """
        return self._trajectory.xyz * 10
    
    def __str__(self):
        return self._trajectory.__str__()

    def __repr__(self):
        return self._trajectory.__repr__()
    def __len__(self):
        return len(self._trajectory)

    def to_resmol(self):
        """
        back to RestrainedMolecule
        - trajectory frames into conformers
        - record restrains
        """
        raise NotImplementedError

    def rerun(self, overwrite = False):
        """
        rerun the simulation with the same settings contained in this object
        overwrites the `trajectory` attribute
        """
        raise NotImplementedError