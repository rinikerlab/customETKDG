import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
import sys
import difflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def assign_hydrogen_pdbinfo(mol):
    """
    assumes all heavy atoms have complete PDB information
    """
    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetPDBResidueInfo() is None and atm.GetAtomicNum() == 1:
            assert len(atm.GetNeighbors()) == 1, "String hydrogen with hypervalence at index {}".format(idx)
            heavy_atm = atm.GetNeighbors()[0]

            # print(heavy_atm.GetPDBResidueInfo().GetName().strip())
            numH = 0
            hasMonInfo = 0
            for ngh_atm in heavy_atm.GetNeighbors():
                if ngh_atm.GetAtomicNum() == 1:
                    numH += 1
                    if ngh_atm.GetMonomerInfo() != None:
                        hasMonInfo += 1

            # if numH == 2 and heavy_atm.GetAtomicNum() == 6:  # standard nomenclature HB3 and HB2 at C atoms NOT FOLLOWD
            #    id = str(3 - hasMonInfo)
            if numH >= 2:
                id = str(hasMonInfo + 1)
            else:
                id = ''

            mi = Chem.AtomPDBResidueInfo()
            # tmp = "H{}".format(heavy_atm.GetPDBResidueInfo().GetName().strip())
            tmp = heavy_atm.GetPDBResidueInfo().GetName().strip()
            if tmp != 'CN':
                tmp = tmp.replace('C', '', 1)  # rm C
                if tmp == '':
                    tmp = 'C'
                tmp = "H{}".format(tmp)
                # print(tmp)
                tmp = tmp.replace('N', '', 1)
                tmp = tmp + id
                if tmp == 'H':
                    tmp = 'HN'  # H is called HN in NOE files for clarity
            else:
                tmp = "H{}".format(tmp)
                tmp = tmp + id
            # print(tmp)
            mi.SetName("{: <4s}".format(tmp))  # spacing needed so that atom entries in the output pdb file can be read
            mi.SetIsHeteroAtom(False)

            mi.SetResidueNumber(heavy_atm.GetPDBResidueInfo().GetResidueNumber())
            mi.SetResidueName(heavy_atm.GetPDBResidueInfo().GetResidueName())
            mi.SetChainId(heavy_atm.GetPDBResidueInfo().GetChainId())
            atm.SetMonomerInfo(mi)
    return mol


def map_mol_with_noe(mol, df, verbose):
    """
    The goal is to make the atom names in the df to be
    exactly the same as those in the mol naming

    mol: must already have named hydrogens added!!!
    """
    mol_atom_names2noe_atom_names = {}
    mol_atom_names2atom_index = {}
    mol_resid_dict = {}
    noe_resid_dict = {}

    for idx, atm in enumerate(mol.GetAtoms()):  # find all H per residue in PDB mol
        if atm.GetAtomicNum() == 1:  # only hydrogens are relevant for NOE mapping
            key = atm.GetPDBResidueInfo().GetResidueNumber()
            val = atm.GetPDBResidueInfo().GetName().strip()
            mol_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries

    for _, row in df.iterrows():  # find all H per residue in NOE representation
        key = row["Residue_index_1"]
        val = row["Residue_name_1"]
        noe_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries
        key = row["Residue_index_2"]
        val = row["Residue_name_2"]
        noe_resid_dict.setdefault(key, {})[val] = 1  # no duplicate entries

    # matching and splitting of @ cases
    exp_noe_names2mol_atom_names = {}
    comb_noe_names2exp_noe_names = {}
    print('#' * 80)
    print('Atom mapping:')
    print('#' * 80)
    for resid in noe_resid_dict:
        print(f'Residue: {resid}')
        mol_atoms = list(mol_resid_dict.get(resid))
        noe_atoms = list(noe_resid_dict.get(resid))
        for atm in noe_atoms:
            # print(atm, process.extract(atm, mol_atoms, scorer=fuzz.ratio))
            # fit = process.extractOne(atm, mol_atoms, scorer=fuzz.ratio)
            noe_key = (resid, atm)
            if atm in mol_atoms:  # exact match found, just add it.
                mol_val = (resid, atm)
                # mol_atoms.remove(atm)  # already matched, no longer needed
                exp_noe_names2mol_atom_names.setdefault(noe_key, {})[mol_val] = 1
                comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_key] = 1
                print(f'NOE atom {atm} exactly mapped to PDB atom {atm}.')
                added = True
            elif '@' in atm:  # deal with multiplicities
                strip_atm = atm.replace('@', '')
                # print(atm, process.extract(strip_atm, mol_atoms, scorer=fuzz.ratio, limit=7))
                fit = process.extract(strip_atm, mol_atoms, scorer=fuzz.ratio, limit=7)
                try:
                    best_score = fit[0][1]
                except:
                    continue
                mult = 0
                exp_noes = []
                for score in fit:
                    if (score[1] == best_score) and (strip_atm in score[0]):  # add all equivalently well matching names
                        mult += 1
                        noe_val = (resid, score[0])
                        exp_noes.append(score[0])
                        comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_val] = 1
                        exp_noe_names2mol_atom_names.setdefault(noe_val, {})[noe_val] = 1
                        # mol_atoms.remove(score[0])
                if mult is not 0:
                    print(f'NOE atom {atm} expanded to {exp_noes} and mapped to corresponding PDB atoms.')
                else:
                    print(f'NOE atom {atm} could not be expanded and mapped to PDB atoms.')
            else:
                try:  # there might be no match
                    mol_atm = process.extractOne(atm, mol_atoms, scorer=fuzz.ratio, score_cutoff=66)[0]
                    mol_val = (resid, mol_atm)
                    comb_noe_names2exp_noe_names.setdefault(noe_key, {})[noe_key] = 1
                    exp_noe_names2mol_atom_names.setdefault(noe_key, {})[mol_val] = 1
                    print(f'NOE atom {atm} approximately mapped to PDB atom {mol_atm}.')
                except:
                    print(f'Unaccounted NOE atom: {atm}.')
                # print(mult)
            # exp_noe_names2mol_atom_names
        #print(f'Remaining PDB atoms: {mol_atoms}')
        print('#' * 80)

    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetAtomicNum() != 1:  # currently only needs the hydrogen
            continue
        key = (
            int(format(atm.GetPDBResidueInfo().GetResidueNumber())),
            "{}".format(atm.GetPDBResidueInfo().GetName().strip()))
        mol_atom_names2noe_atom_names[key] = set()
        mol_atom_names2atom_index.setdefault(key, []).append(idx)

    noe_atom_pair2upper_distance = dict()
    for _, row in df.iterrows():
        # achieve canonical ordering, so each tuple only needs to be added once
        tup1 = (int(row["Residue_index_1"]), "{}".format(row["Residue_name_1"]))
        tup2 = (int(row["Residue_index_2"]), "{}".format(row["Residue_name_2"]))
        key = (int(tup1[0]), key)  # add back the index
        tup_list = [tup1, tup2]
        tups = sorted(tup_list, key=lambda element: (element[0], element[1]))

        # only keep the most restrictive value
        try:
            old_val = noe_atom_pair2upper_distance[(tups[0], tups[1])]
            noe_atom_pair2upper_distance[(tups[0], tups[1])] = min(old_val, row["Upper_bound_[A]"])
        except KeyError:
            noe_atom_pair2upper_distance[(tups[0], tups[1])] = row["Upper_bound_[A]"]

    # in this case usually go and change the NOE dataframe,
    # because the atom names in the pdb file can actually have meaning when running MD
    trigger = False
    for key, val in mol_atom_names2noe_atom_names.items():
        if len(val) > 1:
            print("Non-unique mapping between PDB atom {} and NOE atoms {}.".format(key, val))
            pick = query_yes_no("Pick most probable NOE atom {} (yes) or exit (no)?"
                                .format(difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]))
            if pick:
                val = difflib.get_close_matches(key[1], [j[1] for j in val], 1, 0.5)[0]
                val = {(key[0], val)}  # add back the index
                # mol_atom_names2noe_atom_names.pop(key, None) # remove non-unique
                mol_atom_names2noe_atom_names[key] = val  # re-add most probable
                print("Chosen mapping: {}   {}".format(key, val))
                exp_noe_names2mol_atom_names[min(val)] = key
            else:
                raise ValueError("Non Unique Mapping(s)")

    if trigger: raise ValueError("Non Unique Mapping(s)")

    if verbose:
        print('#' * 80)
        print("Summary of chosen matches:")
        resid = 0
        for key, val in mol_atom_names2noe_atom_names.items():
            if int(key[0]) is not resid:
                print('#' * 80)
                print(f"Residue {key[0]}")
                resid = int(key[0])

            print(f"PDB name {key} is matched to NOE name {val}.")

        print('#' * 80)
        print('#' * 80)

        for key, val in mol_atom_names2noe_atom_names.items():
            # if int(key[0]) is not resid:
            #    print('#' * 80)
            #    print(f"Residue {key[0]}")
            #    resid = int(key[0])

            print(f"NOE name {key} is matched to PDB name {val}.")

    return mol_atom_names2atom_index, comb_noe_names2exp_noe_names, exp_noe_names2mol_atom_names, \
           noe_atom_pair2upper_distance, df


def parse_xplor(file_path, xpl_regex=r'\w[^ ][\d#\*]+|\w+(?= and)|\w\w?(?= +\))|\w\w?(?=\))'):
    """
    Parses an XPLOR-file and returns a dataframe with NOE values. Only considers lines starting with 'assign ('.
    Distances must be in Angstrom.
    :param file_path: XPLOR text file path
    :param regex: regex to extract 'Residue_index_1', 'Residue_name_1', 'Residue_index_2', 'Residue_name_2',
    'Prob_bound_[A]', 'Low_diff_[A]', 'High_diff_[A]' from standard line format {e.g. assign (residue 2 and name HB3)
    (residue 2 and name HB2) 2.25 0.45 0.45}. Changeable if needed.
    :return: dataframe with distance data in Angstrom
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
            print('Could not recognize XPLOR or BIOSYM formats in file. Attempting to parse \"blank\".')
            blank = True
        assert xplor ^ biosym ^ blank

    with open(file_path, 'rt') as NOE_file:
        if xplor:
            for line in NOE_file:
                line = line.strip()  # remove leading whitespace
                if line.startswith("assign"):  # data follows
                    res = xpl.findall(line)
                    assert len(res) == 7, f'Expected to get 7 entries from XPLOR line. Got {len(res)} from \"{line}\".'
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
    print('#' * 80)

    if xplor:
        print('Parser: Read data as XPLOR format.')
        cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                'Residue_name_2', 'Prob_bound_[A]', 'Low_diff_[A]', 'High_diff_[A]']
        df = pd.DataFrame(data, columns=cols)
        df['Prob_bound_[A]'] = df['Prob_bound_[A]'].astype(float)
        df['Low_diff_[A]'] = df['Low_diff_[A]'].astype(float)
        df['High_diff_[A]'] = df['High_diff_[A]'].astype(float)
        df['Upper_bound_[A]'] = df['Prob_bound_[A]'] + df['High_diff_[A]']
        df['Lower_bound_[A]'] = df['Prob_bound_[A]'] - df['Low_diff_[A]']

    if biosym:
        print('Parser: Read data as BIOSYM format.')
        cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                'Residue_name_2', 'Lower_bound_[A]', 'Upper_bound_[A]']
        df = pd.DataFrame(data, columns=cols)
        df['Upper_bound_[A]'] = df['Upper_bound_[A]'].astype(float)
        df['Lower_bound_[A]'] = df['Lower_bound_[A]'].astype(float)

    if blank:
        print('Parser: Read data as \"blank\" format.')
        cols = ['Residue_index_1', 'Residue_name_1', 'Residue_index_2',
                'Residue_name_2', 'Upper_bound_[A]']
        df = pd.DataFrame(data, columns=cols)
        df['Upper_bound_[A]'] = df['Upper_bound_[A]'].astype(float)

    df['Residue_index_1'] = df['Residue_index_1'].astype(int)
    df['Residue_index_2'] = df['Residue_index_2'].astype(int)

    return df


def parse_csv(file_path, unit, sep="\s", comment="#"):
    assert (unit == 'A' or unit == 'nm')

    df = pd.read_csv(file_path, sep=sep, comment=comment)

    if unit == 'nm':
        df['Upper_bound_[A]'] = df['Upper_bound_[nm]'] * 10  # nm to A conversion
    return df


def sanitize_xplor_noe(noe_df):
    isinstance(noe_df, pd.DataFrame)

    noe_df = noe_df.applymap(lambda s: s.upper() if type(s) == str else s)  # make strings uppercase
    noe_df = noe_df.replace(["\*", "#"], "@", regex=True)  # convert from e.g. XPLOR format to GROMOS convention
    noe_df = noe_df.applymap(lambda s: 'H{}@'.format(s.replace('Q', '', 2)) if (type(s) == str and 'Q' in s) else s)

    return noe_df


def sanitize_noe(noe_df):
    isinstance(noe_df, pd.DataFrame)

    noe_df = noe_df.applymap(lambda s: s.upper() if type(s) == str else s)  # make strings uppercase
    noe_df = noe_df.replace(["\*", "#"], "@", regex=True)  # convert from e.g. XPLOR format to GROMOS convention

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


def get_noe_restraint_bmat(mol, df, verbose=False, up_tol=1):
    tot, err = 0, 0
    rdmol_NOE = []
    mol_atom_names2atom_index, comb_noe_names2exp_noe_names, exp_noe_names2mol_atom_names, \
    noe_atom_pair2upper_distance, df = map_mol_with_noe(mol, df, verbose)
    bmat = AllChem.GetMoleculeBoundsMatrix(mol, useMacrocycle14config=True)
    # print("mol_atom_names2atom_index:")
    # print(mol_atom_names2atom_index)
    for key, val in noe_atom_pair2upper_distance.items():
        tot = tot + 1
        #try:
        if (key[0] in comb_noe_names2exp_noe_names.keys()) and key[1] in comb_noe_names2exp_noe_names.keys():
            mol_atm_1 = list(exp_noe_names2mol_atom_names[list(comb_noe_names2exp_noe_names[key[0]])[0]])[0]
            mol_atm_2 = list(exp_noe_names2mol_atom_names[list(comb_noe_names2exp_noe_names[key[1]])[0]])[0]
            a = mol_atom_names2atom_index[mol_atm_1]
            b = mol_atom_names2atom_index[mol_atm_2]
            rdmol_NOE.append([a[0], mol_atm_1[0], mol_atm_1[1], key[0][1], b[0], mol_atm_2[0], mol_atm_2[1], key[1][1], val*up_tol])
            for item in product(a, b):
                if bmat[max(item), min(item)] > val*up_tol:  # updates the lower bound only if it is smaller than NOE suggests
                    bmat[max(item), min(item)] = val*up_tol
                # print(item, bmat[min(item), max(item)], val)
                #if bmat[min(item), max(item)] > val*up_tol:  # TODO: desired behavior?
                bmat[min(item), max(item)] = val*up_tol  # upper bound?
        elif key[0] not in comb_noe_names2exp_noe_names.keys():
            print(f'Skipping NOE restraint because {key[0]} not mapped.')
            err = err + 1
        else:
            print(f'Skipping NOE restraint because {key[1]} not mapped.')
            err = err + 1

    rdmol_NOE = pd.DataFrame(data=rdmol_NOE)
    rdmol_NOE.columns = ['Atm_idx_1', 'Residue_1', 'PDB_name_1', 'NOE_name_1', 'Atm_idx_2', 'Residue_2', 'PDB_name_2',
                         'NOE_name_2', 'Dist_[A]']

    print('#' * 80)
    print("Set {} NOE restraints, not able to assign {} restraints.".format(tot - err, err))
    print('#' * 80)
    return bmat, rdmol_NOE


def query_yes_no(question, default="yes"):
    """
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
