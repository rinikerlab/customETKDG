import numpy as np
from itertools import product
# from rdkit.Chem import PandasTools
from . import PandasTools #FIXME siwtch back to rdkit after update
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import logging
import re
from .molecule import *
from .mol_ops import *

logger = logging.getLogger(__name__)

class NOE: 
    """
    could allow only specifying atom indices pairs, for general molecules

    have a schema?

    - save the distances with units
    
    ResAtm1 ResAtm2 RawLowerDistance RawUpperDistance DistanceUsed Tolerance

    """
    def __init__(self):
        # super().__init__()
        self.noe_table = None
        self.display_colnames = [
            "Atom Index", 
            "Atom Name",
            "Residue Number",
            "Residue Name",
        ]
        self.display_df = pd.DataFrame(columns = self.display_colnames)

    # @property
    # def _constructor(self): #for inheriting pandas
    #     return NOE

    def from_dataframe(self, df, which_cols = None, include_lower_bounds = False):
        """Create a NOE datatable from pandas DataFrame. The minimum needed
        information to create the NOE:

        cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                'Residue_name_2', 'Upper_bound_[A]']

        if include_lower_bounds is True, then the colum Lower_bound_[A] also
        must be present

        Parameters
        ----------
        df : pd.DataFrame

        which_cols : list of indices, optional The columns to obtain the
            information, by default None

        include_lower_bounds : use NOE lower bounds, by default False
        """
        cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                'Residue_name_2', 'Upper_bound_[A]']
        if include_lower_bounds:
            cols.append('Lower_bound_[A]')
        
        if which_cols is None:
            df = df.loc[:, cols]
        else:
            assert len(which_cols) == len(cols), ValueError(f"{len(cols)} Columns are needed from input dataframe.")
            df = df.iloc[:, which_cols]
            df.columns = cols
            
        df['Upper_bound_[A]'] = df['Upper_bound_[A]'].astype(float)
        df['Residue_index_1'] = df['Residue_index_1'].astype(int)
        df['Residue_index_2'] = df['Residue_index_2'].astype(int)
        df['Residue_name_1'] = df['Residue_name_1'].astype(str)
        df['Residue_name_2'] = df['Residue_name_2'].astype(str)
        if include_lower_bounds:
            df['Lower_bound_[A]'] = df['Lower_bound_[A]'].astype(float)

        self.noe_table =  self.sanitize_xplor_noe(df)

        self.sanity_check()


    # @classmethod
    def from_explor(self, file_path, xpl_regex=r'\w[^ ][\d#\*]+|\w+(?= and)|\w\w?(?= +\))|\w\w?(?=\))'): #TODO how the lower bound is calculated?
        """
        Parses an XPLOR-file and returns a dataframe with NOE values. Only considers lines starting with 'assign ('.
        Distances must be in Angstrom.

        Parameters
        ------------
        file_path: str
            XPLOR text file path
        xpl_regex: regex 
            regex to extract 'Residue_index_1', 'Residue_name_1', 'Residue_index_2', 'Residue_name_2',
            'Prob_bound_[A]', 'Low_diff_[A]', 'High_diff_[A]' from standard line format {e.g. assign (residue 2 and name HB3)
            (residue 2 and name HB2) 2.25 0.45 0.45}. Changeable if needed.

        Returns
        -------------
        dataframe with distance data in Angstrom
        """
        old_xpl_reg = r'\w[^ ][\d#\*]+|\w+(?= and)|\w\w?(?= +\))'
        xpl = re.compile(xpl_regex)
        bio = re.compile(r'[^00]\.\w00|(?<=\w\w:)[^ ]+|(?<=_)\d+')
        data = []
        xplor, biosym, blank = False, False, False

        #  detect format
        with open(file_path, 'rt') as NOE_file:
            for line in NOE_file:
                line = line.strip()  # remove leading whitespace
                if line.startswith("assign"):  # XPLOR standard format
                    xplor = True
                if line.startswith('1:'):  # BIOSYM line for strand 1
                    biosym = True
            if xplor and biosym:
                raise Exception('Recognized XPLOR and BIOSYM formats in file. Could not parse.')
            if (not xplor) and (not biosym):
                logger.warning('Could not recognize XPLOR or BIOSYM formats in file. Attempting to parse \"blank\".')
                blank = True
            assert xplor ^ biosym ^ blank

        with open(file_path, 'rt') as NOE_file:
            if xplor:
                for idx, line in enumerate(NOE_file):
                    line = line.strip()  # remove leading whitespace
                    if line.startswith("assign"):  # data follows
                        line = line.split("!")[0]
                        line = line.replace("\t", " ")
                        line = re.sub(r'\s+',' ',line)
                        res = xpl.findall(line)
                        assert len(res) == 7, f'Expected to get 7 entries from XPLOR line. Got {len(res)} from \"{line}\" on line {idx + 1}.'
                        data.append(res)
                    if line.startswith("# Ambiguous"):  # uncertain NOEs follow, stop
                        break
                    if line.startswith("!dihedral restraints") or line.startswith("!HBonds"):  # other data, stop
                        break
            if biosym:
                for line in NOE_file:
                    line = line.strip()  # remove leading whitespace
                    if line.startswith('1:'):  # data follows
                        res = bio.findall(line)
                        assert len(res) == 6, f'Expected to get 6 entries from BIOSYM line. Got {len(res)} from \"{line}\".'
                        data.append(res)
            if blank:
                for line in NOE_file:
                    line = line.split('#', 1)[0]  # discard comments
                    line = line.strip()  # remove leading whitespace
                    if line.startswith("*"):  # comments
                        continue
                    if line.startswith("Entry"):  # models start, NOE data must come before
                        break
                    if line is not '':
                        res = line.split()
                        res.pop(1)
                        res.pop(3)
                        assert len(res) == 5, f'Expected to get 5 entries from blank line. Got {len(res)} from \"{line}\".'
                        data.append(res)

        assert len(data) is not 0,  'No NOE data recognized. Could not parse.'  # could not parse, stop

        if xplor:
            logger.info('Parser: Read data as XPLOR format.')
            cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                    'Residue_name_2', 'Prob_bound_[A]', 'Low_diff_[A]', 'High_diff_[A]']
            df = pd.DataFrame(data, columns=cols)
            df['Prob_bound_[A]'] = df['Prob_bound_[A]'].astype(float)
            df['Low_diff_[A]'] = df['Low_diff_[A]'].astype(float)
            df['High_diff_[A]'] = df['High_diff_[A]'].astype(float)
            df['Upper_bound_[A]'] = df['Prob_bound_[A]'] + df['High_diff_[A]']
            df['Lower_bound_[A]'] = df['Prob_bound_[A]'] - df['Low_diff_[A]']

        if biosym:
            logger.info('Parser: Read data as BIOSYM format.')
            cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                    'Residue_name_2', 'Lower_bound_[A]', 'Upper_bound_[A]']
            df = pd.DataFrame(data, columns=cols)
            df['Upper_bound_[A]'] = df['Upper_bound_[A]'].astype(float)
            df['Lower_bound_[A]'] = df['Lower_bound_[A]'].astype(float)

        if blank:
            logger.info('Parser: Read data as \"blank\" format.')
            cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                    'Residue_name_2', 'Upper_bound_[A]']
            df = pd.DataFrame(data, columns=cols)
            df['Upper_bound_[A]'] = df['Upper_bound_[A]'].astype(float)

        df['Residue_index_1'] = df['Residue_index_1'].astype(int)
        df['Residue_index_2'] = df['Residue_index_2'].astype(int)

        self.noe_table =  self.sanitize_xplor_noe(df)

        self.sanity_check()

    # @classmethod
    def sanitize_xplor_noe(self, noe_df):
        """Sanitize the parsed XPLOR type NOE data.

        Parameters
        ----------
        noe_df : pd.DataFrame
            DataFrame containing the necessary NOE columns.

        Returns
        -------
        pd.DataFrame
            The inputted noe_df.
        """
        isinstance(noe_df, pd.DataFrame)

        noe_df = noe_df.applymap(lambda s: s.upper() if type(s) == str else s)  # make strings uppercase
        noe_df = noe_df.replace([r"\*", "#"], "@", regex=True)  # convert from e.g. XPLOR format to GROMOS convention
        noe_df = noe_df.applymap(lambda s: 'H{}@'.format(s.replace('Q', '', 2)) if (type(s) == str and 'Q' in s) else s)

        return noe_df


    def sanitize_noe(self, noe_df):  #XXX deprecated
        isinstance(noe_df, pd.DataFrame)

        noe_df = noe_df.applymap(lambda s: s.upper() if type(s) == str else s)  # make strings uppercase
        noe_df = noe_df.replace([r"\*", "#"], "@", regex=True)  # convert from e.g. XPLOR format to GROMOS convention

        # TODO: Manipulate distance restraints
        # divide @@ cases into two methyl groups
        # TODO: problem with rings
        new_rows = []
        for idx, row in noe_df.iterrows():
            if (row["Residue_name_1"] == "HD@@"):
                add_row = row.copy()
                row["Residue_name_1"] = "HD1"
                add_row["Residue_name_1"] = "HD2"
                noe_df.drop(idx, inplace=True)
                new_rows.append(row)
                new_rows.append(add_row)
        for idx, row in noe_df.iterrows():
            if (row["Residue_name_2"] == "HD@@"):
                add_row = row.copy()
                row["Residue_name_2"] = "HD1"
                add_row["Residue_name_2"] = "HD2"
                noe_df.drop(idx, inplace=True)
                new_rows.append(row)
                new_rows.append(add_row)
        noe_df = noe_df.append(new_rows)

        return noe_df
        
    def _indices_from_noe_table_row(self, row, mol_frame, hydrogen_dict, messages):
        """Determine the atom indices corresponding to entries in the NOE datatable."""
        if row["Residue_name_1"].endswith("@"):
            idx1 =  mol_frame[
                (mol_frame["resid"] == row["Residue_index_1"]) &
                (mol_frame["name"].str.startswith(row["Residue_name_1"].strip("@")))
            ]["index"].tolist()
        else:
            idx1 =  mol_frame[
                (mol_frame["resid"] == row["Residue_index_1"]) &
                (mol_frame["name"] == row["Residue_name_1"])
            ]["index"].tolist()
        if len(idx1) == 0:
            messages.add("Residue {} atom {} in NOE table is not found in the corresponding residue in molecule containing: {}".format(row["Residue_index_1"], row["Residue_name_1"], hydrogen_dict[int(row["Residue_index_1"])]))

        if row["Residue_name_2"].endswith("@"):
            idx2 =  mol_frame[
                (mol_frame["resid"] == row["Residue_index_2"]) &
                (mol_frame["name"].str.startswith(row["Residue_name_2"].strip("@")))
            ]["index"].tolist()
        else:
            idx2 =  mol_frame[
                (mol_frame["resid"] == row["Residue_index_2"]) &
                (mol_frame["name"] == row["Residue_name_2"])
            ]["index"].tolist()
        if len(idx2) == 0:
            messages.add("Residue {} atom {} in NOE table is not found in the corresponding residue in molecule containing: {}".format(row["Residue_index_2"], row["Residue_name_2"], hydrogen_dict[int(row["Residue_index_2"])]))
        return idx1, idx2

    def _mol2df(self, mol):
        """Create a mol datatable containing the atom index, residue number and atom name of all hydrogens in molecule."""
        hydrogen_dict = hydrogen_dict_from_pdbmol(mol)
        mol_data = [(idx, atm.GetPDBResidueInfo().GetResidueNumber() , atm.GetPDBResidueInfo().GetName().strip()) for idx, atm in enumerate(mol.GetAtoms()) if atm.GetAtomicNum() == 1] #XXX insofar only hydrogens, but in the future?
        return hydrogen_dict, pd.DataFrame.from_records(mol_data, columns = ["index", "resid", "name"])

    def add_noe_to_mol(self, mol, remember_chemical_equivalence = False): #XXX as distance bounds, reflect this in name?
        #XXX can inform which NOEs are missing 
        """
        Add the NOE datatable information to the corresponding molecule object for NOE-guided conformer generation.

        Returns
        ---------------
        mol : RestrainedMolecule
            Mol object containing info about which hydrogen pairs have NOE restraints.
        remember_chemical_equivalence: bool
            keeps an list of integers equal in size to the upper bound distance dataframe, if two adjacent rows are chemically equivalent, the same integer is assigned in list. 
        """
        # self.tolerance
        #XXX check that the atom pairs are both hydrogens (in the future can be other elements)

        hydrogen_dict, mol_frame = self._mol2df(mol)

        upper_df = []
        lower_df = []

        messages = set([])

        if remember_chemical_equivalence:
            counter = 0
            self.chemical_equivalence_list = []

        for _, row in self.noe_table.iterrows():
            idx1, idx2 = self._indices_from_noe_table_row(row, mol_frame, hydrogen_dict, messages)

            if len(idx1) > 0 and len(idx2) > 0:
                val  = row["Upper_bound_[A]"] #TODO make this changeable
                if "Lower_bound_[A]" in row:
                    lower_val = row["Lower_bound_[A]"]
                else:
                    lower_val = None
                for a,b in product(idx1, idx2):
                    upper_df.append((a,b,val))
                    if lower_val is not None:
                        lower_df.append((a,b,lower_val))
                    if remember_chemical_equivalence:
                        self.chemical_equivalence_list.append(counter)
                if remember_chemical_equivalence:
                    counter += 1
        
        if len(messages):
            logger.warning("\n".join(messages))
        
        
        
        upper_df = pd.DataFrame.from_records(upper_df, columns = ["idx1", "idx2", "distance"])  #FIXME which distance specify?
        if not remember_chemical_equivalence:
            upper_df.sort_values(by = ["idx1", "idx2"], inplace = True, ignore_index = True) #XXX ordering is important when tracking chemical equivalence
        if lower_df:
            lower_df = pd.DataFrame.from_records(lower_df, columns = ["idx1", "idx2", "distance"])  #FIXME which distance specify?
            if not remember_chemical_equivalence:
                lower_df.sort_values(by = ["idx1", "idx2"], inplace = True, ignore_index = True) #XXX ordering is important when tracking chemical equivalence
        
        mol = RestrainedMolecule(mol)
        mol.distance_upper_bounds = upper_df
        if lower_df is not []:
            mol.distance_lower_bounds = lower_df
        else:
            mol.distance_lower_bounds = None
        return mol


    def check_match_to_mol(self, mol):  
        """Check the matching between the NOE datatable and a molecule object, record statistics about the NOEs, e.g.:
        - how many bonds apart
        - add columns to the NOE table
        - num bonds away
        - num residues away  this is per pair

        Results in a datatable highlighting the parent heavy atom as a fragment  this is per each unique parent atom appeared in NOE.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule to be matched to the NOE records.

        Returns
        -------
        pd.DataFrame    
            A datatable highlighting all the hydrogens that have NOE record associated to them.

        """
        hydrogen_dict, mol_frame = self._mol2df(mol)
        bonds_apart, residues_apart = [],[]
        matched = [True] * self.noe_table.shape[0]
        dmat = Chem.GetDistanceMatrix(mol)

        #always zero this df upon calling
        self.display_df = pd.DataFrame(columns = self.display_colnames)

        messages = set([])
        for ri, row in self.noe_table.iterrows():
            idx1, idx2 = self._indices_from_noe_table_row(row, mol_frame, hydrogen_dict, messages)
            if len(idx1) > 0 and len(idx2) > 0:
                bonds_apart.append(int(dmat[idx1[0], idx2[0]]))

                tmp = {mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueNumber() for i in Chem.GetShortestPath(mol, idx1[0], idx2[0])}
                residues_apart.append(len(tmp) - 1)

                self.display_df = self.display_df.append(pd.DataFrame(
                    [
                        [",".join(map(str,idx1)), 
                        row["Residue_name_1"],
                        row["Residue_index_1"], 
                        mol.GetAtomWithIdx(idx1[0]).GetPDBResidueInfo().GetResidueName()]
                    ], 
                    columns = self.display_df.columns,
                ),  ignore_index = True)

                self.display_df = self.display_df.append(pd.DataFrame(
                    [
                        [",".join(map(str,idx2)), 
                        row["Residue_name_2"],
                        row["Residue_index_2"],
                        mol.GetAtomWithIdx(idx2[0]).GetPDBResidueInfo().GetResidueName()]
                    ], 
                    columns = self.display_df.columns,
                ),  ignore_index = True)

            else:
                matched[ri] = False
                bonds_apart.append(None)
                residues_apart.append(None)

        self.noe_table["Bonds Separation"] = pd.array(bonds_apart, dtype = pd.Int64Dtype())
        self.noe_table["Residues Separation"] = pd.array(residues_apart, dtype = pd.Int64Dtype())
        self.noe_table["Has Match in Molecule"] = pd.array(matched, dtype = "boolean")

        self.noe_table.sort_values(by = ["Has Match in Molecule", "Residue_index_1", "Residue_index_2"], inplace = True, ignore_index = True)

        #prep for INFO logging
        ring_size = len(get_largest_ring(mol))
        valid_record = self.noe_table["Has Match in Molecule"] == True
        #TODO add info about the kind of distances in the measurement?, or add it after the sanity_check after parsing
        logger.info("""
        {} records assigned to molecule (records for chemically equivalent atoms count as one), including:\n{}
        Largest separation has atoms separated by {} bonds{}         
        {} records cannot be assigned to molecule due to mismatch.
        """.format(
            np.sum(valid_record),
            "\n".join(
                ["               -{} records for atoms separated by {} residues.".format(
                    np.sum(self.noe_table[valid_record]["Residues Separation"] == nr), 
                    nr) for nr in sorted(set(self.noe_table[valid_record]["Residues Separation"]))]),
            np.max(self.noe_table[valid_record]["Bonds Separation"]),
            ", where the largest ring in molecule is of size {}.".format(ring_size) if ring_size > 0 else ".",
            np.sum(~valid_record),
        ))#TODO
        if len(messages):
            logger.warning("\n".join(messages))


        self.display_df.drop_duplicates(subset = self.display_df.columns[:-1], inplace = True)

        self.display_df.sort_values(by = ["Residue Number", "Atom Name"], inplace = True, ignore_index = True)

        images = []
        for _, row in self.display_df.iterrows():
            tmp_mol = copy.copy(mol)
            tmp_mol = Chem.RWMol(tmp_mol)
            to_remove = [atm.GetIdx() for atm in tmp_mol.GetAtoms() if atm.GetPDBResidueInfo().GetResidueNumber() != row["Residue Number"]]
            for i in reversed(to_remove):
                tmp_mol.RemoveAtom(i)
            # tmp_mol = Chem.MolFromSmiles("c1ccccc1")
            idx_arr = map(int, row["Atom Index"].split(","))
            idx_arr = [i - sum([j < i for j in to_remove]) for i in idx_arr]
            # [mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx()] #highlight parent heavy atom
            setattr(tmp_mol, "__sssAtoms", idx_arr) #using tmp_mol.__sssAtoms can cause ambiguity due to the underscores
            images.append(tmp_mol)
        
        tmp_df = self.display_df.copy()
        tmp_df["Depiction"] = images
        PandasTools.RenderImagesInAllDataFrames(images=True)
        return tmp_df

    def remove_unmatched(self):
        """Remove any NOE datatable entires that cannot be matched to a molecule object.

        Returns
        -------
        pd.DataFrame
            NOE Datatable.
        """
        if "Has Match in Molecule" in self.noe_table:
            self.noe_table = self.noe_table[self.noe_table["Has Match in Molecule"]]
        return self.noe_table

    def sanity_check(self):
        """
        Check the noe table has no duplicate entries, including when swapping order of the atom pair.
        """
        tmp = self.noe_table.copy()

        tmp["Residue_index_1"] = self.noe_table["Residue_index_2"]
        tmp["Residue_index_2"] = self.noe_table["Residue_index_1"]
        tmp["Residue_name_1"] = self.noe_table["Residue_name_2"]
        tmp["Residue_name_2"] = self.noe_table["Residue_name_1"]
        length = len(tmp)

        tmp = tmp.append(self.noe_table)

        if tmp.duplicated(subset = ["Residue_name_1", "Residue_name_2", "Residue_index_1", "Residue_index_2"]).any():
            logger.warning("Duplicated Rows \n {}".format(tmp.iloc[:length][tmp.duplicated(subset = ["Residue_name_1", "Residue_name_2", "Residue_index_1", "Residue_index_2"], keep = False).iloc[:length]])) #FIXME

    def show_records(self):
        """Display the datatable containing the NOE information.

        Returns
        -------
        pd.DataFrame
            
        """
        return self.noe_table
    def __repr__(self):
        return self.noe_table.__repr__()

    def manipulate_noe_table(self): #TODO better name
        """changing, scaling values so on
        """
        raise NotImplementedError
    def generate_all_noe_pairs(self): #probably no need
        raise NotImplementedError
    def add_entry(self, ):
        self.sanity_check()
        raise NotImplementedError


# #deprecated
# def map_mol_with_noe(mol, df, verbose):
#     """
#     The goal is to make the atom names in the df to be
#     exactly the same as those in the mol naming

#     mol: must already have named hydrogens added!!!
#     """
#     mol_atom_names2noe_atom_names = {}
#     mol_atom_names2atom_index = {}
#     mol_resid_dict = {}
#     noe_resid_dict = {}

#     for idx, atm in enumerate(mol.GetAtoms()):  # find all H per residue in PDB mol
#         if atm.GetAtomicNum() == 1:  # only hydrogens are relevant for NOE mapping
#             key = atm.GetPDBResidueInfo().GetResidueNumber()
#             val = atm.GetPDBResidueInfo().GetName().strip()
#             mol_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries

#     for _, row in df.iterrows():  # find all H per residue in NOE representation
#         key = row["Residue_index_1"]
#         val = row["Residue_name_1"]
#         noe_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries
#         key = row["Residue_index_2"]
#         val = row["Residue_name_2"]
#         noe_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries

#     # matching and splitting of @ cases
#     exp_noe_names2mol_atom_names = {}
#     comb_noe_names2exp_noe_names = {}
#     print('#' * 80)
#     print('Atom mapping:')
#     print('#' * 80)
#     for resid in noe_resid_dict:
#         print(f'Residue: {resid}')
#         mol_atoms = list(mol_resid_dict.get(resid))
#         noe_atoms = list(noe_resid_dict.get(resid))
#         for atm in noe_atoms:
#             # print(atm, process.extract(atm, mol_atoms, scorer=fuzz.ratio))
#             # fit = process.extractOne(atm, mol_atoms, scorer=fuzz.ratio)
#             noe_key = (resid, atm)
#             if atm in mol_atoms:  # exact match found, just add it.
#                 mol_val = (resid, atm)
#                 # mol_atoms.remove(atm)  # already matched, no longer needed
#                 exp_noe_names2mol_atom_names.setdefault(noe_key, {})[mol_val] = 1
#                 comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_key] = 1
#                 print(f'NOE atom {atm} exactly mapped to PDB atom {atm}.')
#                 added = True
#             elif '@' in atm:  # deal with multiplicities
#                 strip_atm = atm.replace('@', '')
#                 # print(atm, process.extract(strip_atm, mol_atoms, scorer=fuzz.ratio, limit=7))
#                 fit = process.extract(strip_atm, mol_atoms, scorer=fuzz.ratio, limit=7)
#                 try:
#                     best_score = fit[0][1]
#                 except:
#                     continue
#                 mult = 0
#                 exp_noes = []
#                 for score in fit:
#                     if (score[1] == best_score) and (strip_atm in score[0]):  # add all equivalently well matching names
#                         mult += 1
#                         noe_val = (resid, score[0])
#                         exp_noes.append(score[0])
#                         comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_val] = 1
#                         exp_noe_names2mol_atom_names.setdefault(noe_val, {})[noe_val] = 1
#                         # mol_atoms.remove(score[0])
#                 if mult is not 0:
#                     print(f'NOE atom {atm} expanded to {exp_noes} and mapped to corresponding PDB atoms.')
#                 else:
#                     print(f'NOE atom {atm} could not be expanded and mapped to PDB atoms.')
#             else:
#                 try:  # there might be no match
#                     mol_atm = process.extractOne(atm, mol_atoms, scorer=fuzz.ratio, score_cutoff=66)[0]
#                     mol_val = (resid, mol_atm)
#                     comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_key] = 1
#                     exp_noe_names2mol_atom_names.setdefault(noe_key, {})[mol_val] = 1
#                     print(f'NOE atom {atm} approximately mapped to PDB atom {mol_atm}.')
#                 except:
#                     print(f'Unaccounted NOE atom: {atm}.')
#                 # print(mult)
#             # exp_noe_names2mol_atom_names
#         #print(f'Remaining PDB atoms: {mol_atoms}')
#         print('#' * 80)

#     for idx, atm in enumerate(mol.GetAtoms()):
#         if atm.GetAtomicNum() != 1:  # currently only needs the hydrogen
#             continue
#         key = (
#             int(format(atm.GetPDBResidueInfo().GetResidueNumber())),
#             "{}".format(atm.GetPDBResidueInfo().GetName().strip()))
#         mol_atom_names2noe_atom_names[key] = set()
#         mol_atom_names2atom_index.setdefault(key, []).append(idx)

#     noe_atom_pair2upper_distance = dict()
#     for _, row in df.iterrows():
#         # achieve canonical ordering, so each tuple only needs to be added once
#         tup1 = (int(row["Residue_index_1"]), "{}".format(row["Residue_name_1"]))
#         tup2 = (int(row["Residue_index_2"]), "{}".format(row["Residue_name_2"]))
#         key = (int(tup1[0]), key)  # add back the index
#         tup_list = [tup1, tup2]
#         tups = sorted(tup_list, key=lambda element: (element[0], element[1]))

#         # only keep the most restrictive value
#         try:
#             old_val = noe_atom_pair2upper_distance[(tups[0], tups[1])]
#             noe_atom_pair2upper_distance[(tups[0], tups[1])] = min(old_val, row["Upper_bound_[A]"])
#         except KeyError:
#             noe_atom_pair2upper_distance[(tups[0], tups[1])] = row["Upper_bound_[A]"]

#     # in this case usually go and change the NOE dataframe,
#     # because the atom names in the pdb file can actually have meaning when running MD
#     trigger = False
#     for key, val in mol_atom_names2noe_atom_names.items():
#         if len(val) > 1:
#             print("Non-unique mapping between PDB atom {} and NOE atoms {}.".format(key, val))
#             pick = query_yes_no("Pick most probable NOE atom {} (yes) or exit (no)?"
#                                 .format(difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]))
#             if pick:
#                 val = difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]
#                 val = {(key[0], val)}  # add back the index
#                 # mol_atom_names2noe_atom_names.pop(key, None) # remove non-unique
#                 mol_atom_names2noe_atom_names[key] = val  # re-add most probable
#                 print("Chosen mapping: {}   {}".format(key, val))
#                 exp_noe_names2mol_atom_names[min(val)] = key
#             else:
#                 raise ValueError("Non Unique Mapping(s)")

#     if trigger: raise ValueError("Non Unique Mapping(s)")

#     if verbose:
#         print('#' * 80)
#         print("Summary of chosen matches:")
#         resid = 0
#         for key, val in mol_atom_names2noe_atom_names.items():
#             if int(key[0]) is not resid:
#                 print('#' * 80)
#                 print(f"Residue {key[0]}")
#                 resid = int(key[0])

#             print(f"PDB name {key} is matched to NOE name {val}.")

#         print('#' * 80)
#         print('#' * 80)

#         for key, val in mol_atom_names2noe_atom_names.items():
#             # if int(key[0]) is not resid:
#             #    print('#' * 80)
#             #    print(f"Residue {key[0]}")
#             #    resid = int(key[0])

#             print(f"NOE name {key} is matched to PDB name {val}.")

#     return mol_atom_names2atom_index, comb_noe_names2exp_noe_names, exp_noe_names2mol_atom_names, \
#            noe_atom_pair2upper_distance, df
