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

    def __init__(self, structure, smiles = None):
        self.structure = structure
        self.frame_separation = None
        if self.structure.title: #XXX more valid check to see if the supplied smiles actually matches the molecule?
            self.smiles = self.structure.title
        else:
            if smiles is None:
                raise ValueError("Require valid SMILES of the molecule.")
            else:
                self.smiles = smiles
        self._trajectory = None # mdtraj object
    
    def add_frames(self, xyz):
        if self._trajectory is None: #no frames yet
            xyz = np.array(xyz)
            self._trajectory = md.Trajectory(xyz, md.Topology().from_openmm(self.structure.topology))
        else:
            self._trajectory.xyz = np. concatenate([self._trajectory.xyz, xyz])

    def rerun(self, overwrite = False):
        """
        rerun the simulation with the set of settings contained in this object
        overwrites the `trajectory` attribute
        """
        raise NotImplementedError
    
    def to_mol(self):
        """
        trajectory frames into conformers
        """
        raise NotImplementedError

    def to_resmol(self):
        """
        back to RestrainedMolecule
        - trajectory frames into conformers
        - record restrains
        """
        raise NotImplementedError

    
    def __str__(self):
        return self._trajectory.__str__()

    def __repr__(self):
        return self._trajectory.__repr__()
    def __len__(self):
        return len(self._trajectory)