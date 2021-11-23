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
        self.bounds_df_colnames = [
            "idx1"
            "idx2",
            "distance",
        ]
        self._distance_upper_bounds = pd.DataFrame(columns = self.bounds_df_colnames)
        self._distance_lower_bounds = pd.DataFrame(columns = self.bounds_df_colnames)

        self._conformers = None  #XXX needed?
        self._is_minimised = False #TODO same dimension as conformers, 
        self._minimised_conformers = None
        self._is_minimised = np.empty(shape = (0,), dtype = bool)
        self._coordinates = None
        self._picked_conformers = None #cannot be set, need to perserve order
        self.upper_scaling = 1.0 #scaling constant applied to distances
        self.lower_scaling = 1.0 #scaling constant applied to distances
        self.MIN_MACROCYCLE_RING_SIZE = 9
        super().__init__(mol)

    def load_mol(self, smiles, pdb_filename):
        """Create a RDKit molecule by matching a SMILES string (obtain correct bond orders) and a pdb file (obtain atom names).

        Parameters
        ----------
        smiles : str
            SMILES string.
        pdb_filename : str
            Path to pdf file.
        """
        
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
        self.RemoveAllConformers()
        df = df.astype({"idx1" : np.uint, "idx2": np.uint, "distance" : float})
        df.dropna(inplace = True) 
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
    #XXX currently not being used
    def scale_lower_distances(self, scale_value): 
        """Apply a scaling factor to the lower bound distances between atom pairs.
        If scaling is one single value then the same factor is applied to all distances while if a list of values is supplied than different scaling factor can be applied to each distance. 
        ! Currently the lower bound scaling is not being used in the conformer generation process.

        Parameters
        ----------
        scale_value : float or list of floats
            The factor to apply.
        """
        if type(scale_value) in [int, float]:
            scale_value = [scale_value]

        if len(scale_value) == 1: #all the same scaling
            scale_value *= len(self._distance_lower_bounds)
        
        assert len(scale_value) == len(self._distance_lower_bounds), "Number of distance restraints do not match number of scalings."

        self.lower_scaling = scale_value

    #FIXME
    def scale_upper_distances(self, scale_value): 
        """Apply a scaling factor to the upper bound distances between atom pairs.
        If scaling is one single value then the same factor is applied to all distances while if a list of values is supplied than different scaling factor can be applied to each distance. 

        Parameters
        ----------
        scale_value : float or list of floats
            The factor to apply.
        """
        if type(scale_value) in [int, float]:
            scale_value = [scale_value]

        if len(scale_value) == 1: #all the same scaling
            scale_value *= len(self._distance_upper_bounds)
        
        assert len(scale_value) == len(self._distance_upper_bounds), "Number of distance restraints do not match number of scalings."

        self.upper_scaling = copy.deepcopy(scale_value)

    def update_bmat(self, include_lower_bounds=False):  #TODO only upperbound?
        """
        Load the user defined bounds matrix to the RestrainedMolecule object for conformer generation
        """
        self.RemoveAllConformers() #FIXME only do it if bmat is found to change

        self.bmat = AllChem.GetMoleculeBoundsMatrix(self, useMacrocycle14config=True) #FIXME customisable for non-macrocycle

        self.scale_upper_distances(self.upper_scaling)
        self.scale_lower_distances(self.lower_scaling)

        for index, row in self._distance_upper_bounds.iterrows():
            a, b, dist = row

            dist *= self.upper_scaling[index]

            a,b = int(a), int(b) #XXX awkard, even specified as int in dataframe still comes out as python floats
            self.bmat[min((a,b)), max((a,b))] = dist #XXX only change if it shrinks the bounds?
            # if the current lower bound is larger than the upper bound we're adding, shrink it:
            if self.bmat[max((a,b)), min((a,b))] > dist: 
                self.bmat[max((a,b)), min((a,b))] = dist

        if include_lower_bounds and self._distance_lower_bounds is not None:
            for index, row in self._distance_lower_bounds.iterrows():
                a, b, dist = row

                dist *= self.lower_scaling[index]

                a,b = int(a), int(b) #XXX awkard, even specified as int in dataframe still comes out as python floats
                self.bmat[max((a,b)), min((a,b))] = dist #XXX only change if it shrinks the bounds?
                # if the current upper bound is less than the lower bound we're adding, increase it:
                if self.bmat[min((a,b)), max((a,b))] < dist: 
                    self.bmat[min((a,b)), max((a,b))] = dist


        self.triangular_smoothed_bmat = copy.deepcopy(self.bmat) #FIXME split it out?
        DistanceGeometry.DoTriangleSmoothing(self.triangular_smoothed_bmat)


    #######################################
    # Confgen
    #######################################
    def estimate_time(self, max_time_per_conf = 10, repeats = 3):
        """Currently rdkit conformer generation can halt when too strict a bound matrix is provided.
        So it is recommended practise to use this function to first estimate the time cost, as this function does not stall.


        Parameters
        ----------
        max_time_per_conf : int, optional
            Upper time limit in seconds to kill the process of generating **one** conformer, by default 10.
        repeats : int, optional
            The number of repeats for conformer generation to get average, by default 3.

        Returns
        -------
        float
            The averaged time cost to generate a conformer. NaN is returned when time exceeds the `max_time_per_conf` supplied.
        """

        #XXX keep track of how many conformers exist, if process is not finished delete the newly produced conformers?
        out = []
        for _ in range(repeats):
            start = time.time()
            p = multiprocessing.Process(target=self.generate_conformers, args = (1,))
            p.start()
            p.join(max_time_per_conf)
            if p.is_alive():
                logger.error("Conformer Generation process timedout.")
                p.terminate()
                p.join()
                return float("NaN")
            out.append(time.time() - start)
        if len(out):
            logger.info('{:.1f} seconds per conformer on average.'.format(np.mean(out)))
            return np.mean(out)
            
    def generate_conformers(self, num_conf, params = None): 
        """Generate the required number of conformers.

        Parameters
        ----------
        num_conf : int
            The number of conformers to generate
        params : rdkit.EmbedParameters, optional
            EmbedParameters object, by default None, meaning the ETKDGv3 object desgined for macrocycles is used. 
            When a macrocycle is detected, the force scaling is set to 0.3
        """

        #XXX it should have option to use other version of ETKDG as this class is flexible enough to confgen any type of molecule

        #is there some way to forbid EmbedMultipleConf being called on this class object? if this is really necessary
        
        if params is None:
            params = AllChem.ETKDGv3() #FIXME changable 
            params.useRandomCoords = True #TODO changeable
            params.verbose = False #XXX as argument?
            # params.maxAttempts = 0
            params.numThreads = 0 #TODO changeable

            if len(get_largest_ring(self)) >= self.MIN_MACROCYCLE_RING_SIZE:
                logger.info("Molecule is macrocycle (largest ring >= 9 atoms), scaling down the bounds matrix force contribution.")
                params.boundsMatForceScaling = 0.3  

        params.clearConfs = False #XXX allow user to specify this?
        params.SetBoundsMat(self.bmat) #XXX diff to self.bmat, should be triangular smoothed bmat?

        AllChem.EmbedMultipleConfs(self, num_conf, params)

        #keep a record of the etkdg settings once conformers are successfully generated.
        self.etkdg_params = params
        
        # AllChem.AlignMolConformers(self)  #XXX align confs to first for easier visual comparison, No! this will include align to side chain
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
        conf_num = int(conf_num)
        while True: #XXX what if never converge?
            rank = AllChem.MMFFOptimizeMolecule(self, confId = conf_num, maxIters = 100) #XXX threads number to use
            if rank == 0:
                return 
        # while True: #XXX what if never converge?
        #     ranks = AllChem.MMFFOptimizeMoleculeConfs(self, numThreads = 0, maxIters = 100) #XXX threads number to use
            
        #     if np.all([i == 0 for i,j in ranks]):
        #         return [j for i,j in ranks]

    def calculate_energy(self):
        """

        e.g. for ranking conformers
        """
        raise NotImplementedError

    def save_conformers(self, path, confId = -1):
        self._conformers.save(path)

    def save_minimised_conformers(self, confId = -1):
        self._minimsed_conformers.save(path)
    #######################################
    # Selection
    #######################################
    def pick_random(self, num_conf):
        """Randomly pick the required number of conformers.

        Parameters
        ----------
        num_conf : int
            Number of conformers required.

        Returns
        -------
        list
            List of conformer indices.
        """
        out = np.random.choice(self.GetNumConformers(), num_conf, replace = False)
        self._picked_conformers = list([int(i) for i in out])
        return self._picked_conformers

    def pick_diverse(self, num_conf, indices = [], seed = -1): #XXX does not really give unique solutions
        """ Pick a diverse set of conformers based on root mean squared deviation of 3D coordiantes of atoms, optionally only considering diversity on a subset of all atoms.
        This is based on rdkit diverse picker: https://www.rdkit.org/docs/GettingStartedInPython.html?highlight=maccs#picking-diverse-molecules-using-fingerprints
        No unique solution is guaranteed as the pick is stochastic contingent on a random number seed.

        Parameters
        ----------
        num_conf : int
            Number of conformers required.
        indices : list, optional
            Atom indices, by default [], meaning all atom indices are considered.
        seed : int, optional
            Random number seed, by default -1.

        Returns
        -------
        list
            A list of conformer indices.
        """

        AllChem.AlignMolConformers(self, atomIds = indices)
        def distij(i, j):
            return AllChem.GetConformerRMS(self, i, j, atomIds = indices, prealigned = True)
        
        picker = MaxMinPicker()
        out =  picker.LazyPick(distij, self.GetNumConformers(), num_conf, seed = seed)
        # self._picked_conformers = np.array(out) #should not be np array as in the case of pick energy they are tuples
        self._picked_conformers = list(out)
        return self._picked_conformers

    def pick_energy(self, num_conf):
        """Pick conformers with lowest energy, currently only MMFF energies can be calculated.

        Parameters
        ----------
        num_conf : int
            Number of conformers required.

        Returns
        -------
        list
            A list of conformer indices.
        """
        #equires calculating MMFF energies for all conformers, in the process all structures are optimised

        mp = AllChem.MMFFGetMoleculeProperties(self)
        out = [AllChem.MMFFGetMoleculeForceField(self, mp, confId = i).CalcEnergy() for i in range(self.GetNumConformers())] #XXX show progress?
        out = np.array(out) - min(out) #XXX otherwise Boltzmann weight does not work

        self._picked_conformers  = list(np.argsort(out)[:num_conf])
        return [(int(i), out[i]) for i in self._picked_conformers]

    def pick_namfis(self, num_conf, noe, pre_selection = None, tolerance = .0):
        """Pick conformers based on NMR analysis of molecular flexibility in solution (NAMFIS). 
        NAMFIS is a constrained optimisation scheme, starting with assigning all conformer with an equal weight factor (sum of weights for all conformer equals 1), 
        the weights are optimised to give conformer ensemble that obey the NOE measurments (within experimental tolarence).

        Parameters
        ----------
        num_conf : int
            Number of conformers required.
        noe : customETKDG.NOE object
            NOE object containing chemically equivalent hydrogen information.
        pre_selection : list, optional
            A subset of conformer indices in a list from which NAMFIS is run in order to prevent convergence issues, by default None meaning all conformers are considered. 
        tolerance : float, optional
            Allow some overall violation (e.g. due to experimental uncertainty) of specified bounds in the constrained minimisation, by default 0 meaning no tolerance.

        Returns
        -------
        list
            A list of conformer indices that have the highest weights.
        """
        from scipy.optimize import minimize


        MAX_CONF_LIMIT = 200
        if pre_selection is None:
            pre_selection = range(self.GetNumConformers())
        if len(pre_selection) > MAX_CONF_LIMIT:
            logger.warning("Number of conformers exceed {}, NAMFIS might not converge.".format(MAX_CONF_LIMIT))

        distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(self, i) for i in pre_selection])

        df = noe.add_noe_to_mol(self, remember_chemical_equivalence = True).distance_upper_bounds

        distances = distance_matrix_for_each_conformer[:, df.idx1, df.idx2]
        ref_distances = np.array(df.distance)

        # define error scale factor for distances in different ranges
        errors = np.ones(len(ref_distances)) * 0.4
        errors[ref_distances < 6.0] = 0.4
        errors[ref_distances < 3.5] = 0.3
        errors[ref_distances < 3.0] = 0.2
        errors[ref_distances < 2.5] = 0.1    

        #### ce means chemical equivalent
        distances_ce = np.split(distances, np.unique(noe.chemical_equivalence_list, return_index=True)[1][1:], axis = 1) #here I group the distances by their chemical equivalence track
        distances_ce = np.stack([np.mean(d, axis = 1) for d in distances_ce], axis = 1)


        ref_distances_ce = np.split(ref_distances, np.unique(noe.chemical_equivalence_list, return_index=True)[1][1:])
        ref_distances_ce = np.stack([np.mean(d) for d in ref_distances_ce])

        errors_ce = np.split(errors, np.unique(noe.chemical_equivalence_list, return_index=True)[1][1:])
        errors_ce = np.stack([np.mean(d) for d in errors_ce])

        def objective(w): #w is weights
            deviation  = ref_distances_ce - np.average(distances_ce, weights = w, axis = 0)
            deviation /= errors_ce
    #         deviation = np.heaviside(deviation, 0) * deviation #only penalise upper violation
            return np.sum(deviation**2) #squared deviation
    #         return np.linalg.norm(deviation) #square rooted
        
        cons = [{'type':'eq','fun': lambda w: np.sum(w) - 1}] #weights add up to 1
        
        
        cons += [ #does not allow any violation
                {'type':'ineq','fun': lambda w:  (errors_ce + tolerance) - np.absolute(np.average(distances_ce, weights = w, axis = 0) - ref_distances_ce)} 
        ]

    #     cons += [ #does not allow only upper violations
    #                 {'type':'ineq','fun': lambda w: ref_distances_ce - np.average(distances_ce, weights = w, axis = 0) - tolerance} 
    #     ]
        
        weights = np.random.uniform(low = 0, high = 1, size = len(pre_selection)) #uniform weights at start

        out = minimize(
            objective,
            weights, 
            constraints = tuple(cons),
            bounds = tuple((0,1) for _ in range(len(weights))), #each weight constraint
            method='SLSQP')

        if not out["success"]:
            logger.error("NAMFIS failed: {}".format(out["message"]))
            
        weights = out["x"] 

        return list(zip([int(i) for i in np.argsort(-1 * weights)[:num_conf]], weights[np.argsort(weights * -1)[:num_conf]]))

    def pick_least_upper_violation(self, num_conf):
        """Pick conformers that each least violate the NOE

        Parameters
        ----------
        num_conf : int
            Number of conformers required.

        Returns
        -------
        list
            A list of conformer indices.
        """
        distance_matrix_for_each_conformer = np.array([Chem.Get3DDistanceMatrix(self, i) for i in range(self.GetNumConformers())])

        df = self.distance_upper_bounds #FIXME
        distances = distance_matrix_for_each_conformer[:, df.idx1, df.idx2] - np.array(df.distance)
        sum_violations = np.copy(distances)
        sum_violations[sum_violations < 0] = 0
        out = np.sum(sum_violations, axis = 1)
        self._picked_conformers  = list(np.argsort(out)[:num_conf])
        
        return [int(i) for i in self._picked_conformers] #should change picked_conformers to also int type


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
        what sets of information should be maintained?
        """
        raise NotImplementedError

    def __copy__(self):
        newone = super(RestrainedMolecule, self).__copy__()
        newone.__class = self.__class
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        newone = super(RestrainedMolecule, self).__deepcopy__(memo)
        newone.__class__ = self.__class__
        newone.__dict__.update(self.__dict__)
        # cls = self.__class__
        # result = cls.__new__(cls)
        # memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(newone, k, copy.deepcopy(v, memo))
        return newone
    
    # def __str__(self):
    #     #TODO print the number of conformers!
    #     raise NotImplementedError
