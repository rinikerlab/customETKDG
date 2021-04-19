import time
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DistanceGeometry
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import pandas as pd
from .mol_ops import *
import collections 
import logging
import multiprocessing
"""

- EmbedMultipleConfs by defualt overwrites previous rounds of confgen
- 
"""

# ConfGenConfig = collections.namedtuple('ConfGenConfig',
#     ["random_coordinate", ""] "verbose", numThreads, clearConfs
# )


class RestrainedMolecule(Chem.Mol): #XXX name too generic? mention measurements in name?
    """
    Generic class for conformer generation with additional bounds
    such bounds could be distance based, as well as torsional or others (in the future)

    - can accept any indexed molecule format and a set of pairwise indices and a distance to perform confgen
    - store the set of bounds together with conformer ensemble
    - add a mdtraj representation
    """
    def __init__(self, mol):
        self.bounds = None
        self._distance_upper_bounds = None
        self._distance_lower_bounds = None
        self._conformers = None  #XXX needed?
        self._is_minimised = False #TODO same dimension as conformers, 
        self._minimised_conformers = None
        self._is_minimised = np.empty(shape = (0,), dtype = bool)
        self._coordinates = None
        self._picked_conformers = None #cannot be set, need to perserve order
        super().__init__(mol)

    def load_mol(self, smiles, pdb_filename):
        mol = mol_from_smiles_pdb(smiles, pdb_filename)
        self.__init__(mol)

    @property
    def distance_upper_bounds(self):
        return self._distance_upper_bounds

    @distance_upper_bounds.setter
    def distance_upper_bounds(self, df): #FIXME
        #every time this is changed, conformers are cleared
        self.RemoveAllConformers()


        df = df.astype({"idx1" : np.uint, "idx2": np.uint, "distance" : float})
        df.dropna(inplace = True) 
        self._distance_upper_bounds = df

    @property
    def distance_lower_bounds(self):
        return self._distance_lower_bounds

    @distance_lower_bounds.setter
    def distance_lower_bounds(self, df):
        self._distance_lower_bounds = df

    # def add_upper_bounds(self, df): #XXX add_bounds_equivalent_hydrogens = False):
    #     """
    #     what if equivalent hydrogens have different bounds? then they are not equilvanet

    #     """
    #     raise NotImplementedError


    def infer_eccentricity_from_distbounds(self): #XXX should be here? or as itsown function
        """
        see if eccentricity assumption is valid, if so infer the turn of ellipse
        """
        raise NotImplementedError

    #######################################
    # Pre-confgen
    #######################################
    def update_bmat(self):  #TODO only upperbound?
        """
        prepare the bounds matrix into the user specified state
        """
        self.RemoveAllConformers() #FIXME only do it if bmat is found to change

        self.bmat = AllChem.GetMoleculeBoundsMatrix(self, useMacrocycle14config=True) #FIXME customisable for non-macrocycle
        for _, row in self._distance_upper_bounds.iterrows():
            a, b, dist = row
            a,b = int(a), int(b) #XXX awkard, even specified as int in dataframe still comes out as python floats
            self.bmat[min((a,b)), max((a,b))] = dist #XXX only change if strink the bounds?
            if self.bmat[max((a,b)), min((a,b))] > dist: 
                self.bmat[max((a,b)), min((a,b))] = dist

    # def triangular_smooth(self):
    #     """
    #     store bmat before and after smoothing 
    #     identify max changes in distance bounds as some form of `sanity check`
    #     """
        self.triangular_smoothed_bmat = copy.deepcopy(self.bmat) #FIXME split it out?
        DistanceGeometry.DoTriangleSmoothing(self.triangular_smoothed_bmat)


    #######################################
    # Confgen
    #######################################
    def estimate_time(self, max_time_per_conf = 10):
        """ 
        Currently rdkit confgen halts when conformers are not generated

        timeout is the time limit to generate each conformer in second
        """
        #XXX keep track of how many conformers exist, if process is not finished delete the newly produced conformers?
        out = []
        for _ in range(3):
            start = time.time()
            p = multiprocessing.Process(target=self.generate_conformers, args = (1,))
            p.start()
            p.join(max_time_per_conf)
            if p.is_alive():
                logger.error("Conformer Generation process timedout.")
                p.terminate()
                p.join()
                break
            out.append(time.time() - start)
        if len(out):
                logger.info('{:.1f} seconds per conformer on average.'.format(np.mean(out)))


    def generate_conformers(self, num_conf): #XXX it should have option to use other version of ETKDG as this class is flexible enough to confgen any type of molecule
        """
        is there some way to forbid EmbedMultipleConf being called on this class object?
        """
        params = AllChem.ETKDGv3() #FIXME changable 
        params.useRandomCoords = True #TODO changeable
        params.SetBoundsMat(self.bmat) #XXX diff to self.bmat, should be triangular smoothed bmat?
        params.verbose = False #XXX as argument?
        # params.maxAttempts = 0
        params.clearConfs = False
        params.numThreads = 0 #TODO changeable

        AllChem.EmbedMultipleConfs(self, num_conf, params)
        AllChem.AlignMolConformers(self)  #XXX align confs to first for easier visual comparison
        self._is_minimised = np.concatenate((self._is_minimised, np.zeros(num_conf, dtype = bool)))
    #######################################
    # Post-process
    #######################################
    def minimise_conf(self, conf_num = 0):#XXX keep a copy of the conformer prior to minimisation?
        """
        - MMFF
        - openFF
        - xtb 
        - ani2?
        """
        return AllChem.MMFFOptimizeMolecule(self, maxIters = 500, confId = conf_num) #XXX should make other parameters as variable? default iter might be too small

    def calculate_energy(self):
        """

        e.g. for ranking conformers
        """
        raise NotImplementedError

    def simulate(self, confId = 0):
        raise NotImplementedError
    
    def save_conformers(self, path, confId = -1):
        self._conformers.save(path)

    def save_minimised_conformers(self, confId = -1):
        self._minimsed_conformers.save(path)
    #######################################
    # Selection
    #######################################
    def pick_random(self, num_conf):
        self._picked_conformers = np.random.choice(self.GetNumConformers(), num_conf, replace = False)
        return self._picked_conformers

    def pick_diverse(self, num_conf, indices = None, seed = -1): #XXX does not really give unique solutions
        """rdkit diverse picker
        https://www.rdkit.org/docs/GettingStartedInPython.html?highlight=maccs#picking-diverse-molecules-using-fingerprints

        indices None means all atoms

        """
        
        def distij(i, j):
            return AllChem.GetConformerRMS(self, i, j, atomIds = indices, prealigned = True)
        
        picker = MaxMinPicker()
        out =  picker.LazyPick(distij, self.GetNumConformers(), num_conf, seed = seed)
        self._picked_conformers = np.array(out)
        return self._picked_conformers

    def pick_energy(self, num_conf):
        #equires calculating MMFF energies for all conformers, in the process all structures are optimised

        while True: #XXX what if never converge?
            ranks = AllChem.MMFFOptimizeMoleculeConfs(self, numThreads = 0, maxIters = 100)
            
            if np.all([i == 0 for i,j in ranks]):
                self._picked_conformers  = np.argsort([j for i,j in ranks][:num_conf])
                return self._picked_conformers

    def pick_namfis(self, num_conf):

        raise NotImplementedError
        

    #######################################
    # Checks
    #######################################
    def diagnose_missed_close_contacts(self, confId = 0):
        """
        Once conformers have been generated, check for a given conformer 
        whether a close distance is found between atom pairs that should probably have a NOE measurement
        """
        raise NotImplementedError

    def summary(self):
        """
        - report the types of restraints:
            - transannular
            - within-same residue (distinguish sidechain?), e.g. this type of restraints are less important

        for each restraint, report the number of bonds and number of residues separating the atom pair
        """
        raise NotImplementedError

    def _process_info(self):
        """
        Ensures all information is concordant upon any update
        check if some information is missing
        """
        raise NotImplementedError

    #######################################
    # Utils
    #######################################
    @property 
    def conformers(self):
        return self._conformers

    @property 
    def minimised_conformers(self):
        return self._minimised_conformers

    
    def duplicate(self):
        """
        easy stepstone to do confgen on the same molecule
        but with different sets of NOEs
        """
        raise NotImplementedError

    def __add__(self, other):
        """ append the conformers given the same molecule and noe data has been used for conformer generation
        """
        raise NotImplementedError

    def __eq__(self, another):
        """
        ?
        same experimental set of information
        """
        raise NotImplementedError

    def __copy__(self):
        newone = super(Molecule, self).__copy__()
        newone.__class = self.__class
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        newone = super(Molecule, self).__deepcopy__(memo)
        newone.__class__ = self.__class__
        newone.__dict__.update(self.__dict__)
        # cls = self.__class__
        # result = cls.__new__(cls)
        # memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(newone, k, copy.deepcopy(v, memo))
        return newone
