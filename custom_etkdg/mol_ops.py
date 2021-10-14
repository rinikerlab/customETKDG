"""
adapted from my cpeptools package

rdkit mol manipulations
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from functools import reduce
import difflib
import logging
from collections.abc import Iterable
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

# logger.setLevel(logging.INFO)

def load_coordinates(mol, np_array):
    """Load 3D coordinates into rdmol and append them into existing conformer list.

    Parameters
    ----------
    mol : RDKit Mol
        
    np_array : 3D cooridnate
        NxMx3 float arrays, where N is the number of conformers to add, M is the number of atoms in the molecule.

    Returns
    -------
    RDKit Mol
        The inputted molecule with added conformers.
    """
    assert mol.GetNumAtoms() == len(np_array[1]), "number of atoms do not match: {}, {}".format(mol.GetNumAtoms(), len(np_array[1]))
    init = mol.GetNumConformers()
    for idx, coord in enumerate(np_array):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(len(coord)):
            x,y,z = coord[i]
            conf.SetAtomPosition(i, Point3D(x,y,z))
        mol.AddConformer(conf, init + idx)
    return mol


def _select(selection_string): #XXX try to use?
    """
    from our cpeptools package, simple atom group selection semanitcs
    TODOs:
        - include chain name and chain id
    """
    logic_words = {"(", ")", "==", "!=", "and", "or"} #True, False
    rdkit_func_keyword = ".GetPDBResidueInfo()." #FIXME matches better than this?

    selection_string = selection_string.replace("is not", "!=")
    selection_string = selection_string.replace("is", "==")
    selection_string = selection_string.replace("(", " ( ")
    selection_string = selection_string.replace(")", " ) ")

    selection_string = selection_string.lower() #should be after is/ is not replacement

    selection_string = selection_string.replace("resname", "atm.GetPDBResidueInfo().GetResidueName().strip().lower() ")
    selection_string = selection_string.replace("resnum", "str(atm.GetPDBResidueInfo().GetResidueNumber()) ")
    # selection_string = selection_string.replace("index", "str(atm.GetIdx()) ") #headache as there is offset bettwen pdb index and rdkit mol atom index
    selection_string = selection_string.replace("name", "atm.GetPDBResidueInfo().GetName().strip().lower() ")

    selection_string = selection_string.split()
    for idx,val in enumerate(selection_string):
        if val not in logic_words and rdkit_func_keyword not in val:
            selection_string[idx] = "\"{}\"".format(selection_string[idx])
    return "{}".format(" ".join(selection_string))

def select_atoms(mol, selection):
    """Select atoms based on selection semantics.

    Parameters
    ----------
    mol : RDKit Mol.
        
    selection : str
        Selection string.

    Returns
    -------
    List
        List of RDKit Atoms that fit the selection criteria.
    """
    selection = _select(selection)
    return list(eval("filter(lambda atm : {} , mol.GetAtoms())".format(selection)))


def rename(mol, old_name, new_name, resid = "all"):
    """Rename atoms that had a old name to a new name, potentially only for selected residues.

    Parameters
    ----------
    mol : RDKit Mol
        
    old_name : str
        Exact atom name to be replaced.
    new_name : str
        Exact atom name to replace. 
    resid : str of iterable of int, optional
        Perform renaming only one atoms with the provided resid, by default "all", meaning renaming to occur for all residues.

    Returns
    -------
    RDKit Mol
        The inputted mol.

    Raises
    ------
    ValueError
        If the resid is a unrcognised type.
    """
    #TODO checking for resid types

    selection = resid
    if selection == "all":
        selection = set(range(mol.GetNumAtoms() + 1))
    elif isinstance(selection, Iterable) and type(selection) is not str:
        selection = {int(i) for i in selection}
    else:
        raise ValueError("Unrecognized resid type {}".format(type(resid)))
    
    for atm in mol.GetAtoms():
        if atm.GetPDBResidueInfo().GetResidueNumber() in selection and         atm.GetPDBResidueInfo().GetName().strip() == old_name.strip():
            atm.GetPDBResidueInfo().SetName("{: <4s}".format(new_name.strip()))
            # print(atm.GetPDBResidueInfo().GetName())
    return mol


def reorder_atoms(mol):
    """Change index of the atoms to ensure atoms are ordered by ascending residue number.

    Parameters
    ----------
    mol : RDKit Mol
        

    Returns
    -------
    RDKit Mol
        A copy of the inputted mol with new atom order.
    """
    order = [(i.GetPDBResidueInfo().GetName().strip(), i.GetPDBResidueInfo().GetResidueNumber()) for i in mol.GetAtoms()] # currently the atom name is not used in sorting
    order = [i[0] for i in sorted(enumerate(order), key  = lambda x : x[1][1])]
    return Chem.RenumberAtoms(mol, order)

def assign_hydrogen_pdbinfo(mol, hydrogen_dict):
    """Assign atom names for hydrogens based on the heavy atom they are bonded to.
    When there is hydrogen records in the pdb, assumes they are in the same order as the heavy atoms appear in 

    Parameters
    ----------
    mol : RDKit Mol
        
    hydrogen_dict : dict
        The keys are the residue number/id and the values are the list of hydrogen atoms names belonging to each residue.

    Returns
    -------
    RDKit Mol
        The inputted mol with hydrogen atom names.

    Raises
    ------
    ValueError
        If number of hydrogens do not match between the hydrogen dictionary and the molecule.
    """

    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetAtomicNum() > 1: #look for H neighbour of heavy atom
            for atm2 in atm.GetNeighbors():
                if atm2.GetPDBResidueInfo() is None and atm2.GetAtomicNum() == 1:
                    mi = Chem.AtomPDBResidueInfo()

                    try:
                        tmp = hydrogen_dict[atm.GetPDBResidueInfo().GetResidueNumber()].pop(0)
                    except:
                        raise ValueError("SMILES contain more hydrogen than contained in pdb file for residue number {}".format(atm.GetPDBResidueInfo().GetResidueNumber()))
                    logger.info("Residue number {}, hydrogen attached to atom {} is assigned name {}".format(atm.GetPDBResidueInfo().GetResidueNumber(), atm.GetPDBResidueInfo().GetName(), tmp))

                    mi.SetName("{: <4s}".format(tmp))  # spacing needed so that atom entries in the output pdb file can be read
                    mi.SetIsHeteroAtom(False)
                    mi.SetResidueNumber(atm.GetPDBResidueInfo().GetResidueNumber())
                    mi.SetResidueName(atm.GetPDBResidueInfo().GetResidueName())
                    mi.SetChainId(atm.GetPDBResidueInfo().GetChainId())
                    atm2.SetMonomerInfo(mi)
    # AllChem.Compute2DCoords(mol) #XXX why this?
    return mol

def assign_hydrogen_pdbinfo_blind(mol):
    """
    Assign atom names for hydrogens based on the heavy atom they are bonded to. Assumes all heavy atoms have complete PDB information
    Use this function if the pdbfile contains no records of hydrogen name

    Parameters
    ----------
    mol : RDKit Mol
        
    Returns
    -------
    RDKit Mol
        The inputted mol with hydrogen atom names.
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
            logger.info("Residue number {}, hydrogen attached to atom {} is assigned name {}".format(heavy_atm.GetPDBResidueInfo().GetResidueNumber(), heavy_atm.GetPDBResidueInfo().GetName(), tmp)) #TODO make them into one INFO message
    # AllChem.Compute2DCoords(mol) #XXX why this?
    return mol

def hydrogen_dict_from_pdbmol(mol):
    """Generate a dictionary where the keys are the residue numbers and values lists of hydrogen atom names in the molecule.
    This is done by getting the hydrogen names used in the pdb molecule

    Parameters
    ----------
    mol : RDKit Mol

    Returns
    -------
    dict
        Dictionary of hydrogen atom names per each residue.
    """
    out_dict = {}
    for idx, atm in enumerate(mol.GetAtoms()):
        if atm.GetAtomicNum() == 1:
            out_dict.setdefault(atm.GetPDBResidueInfo().GetResidueNumber(), []).append(atm.GetPDBResidueInfo().GetName().strip())
    return out_dict

def copy_stereo(from_mol, to_mol): #TODO add assertions to check molecules are the same?
    """Transfer the stereochemical information from one molecule to another molecule.

    Parameters
    ----------
    from_mol : RDKit Mol
        
    to_mol : RDKit Mol
        

    Returns
    -------
    RDKit Mol
        The inputted `to_mol` with stereo info from `from_mol`.
    """

    match = to_mol.GetSubstructMatch(from_mol)

    # find chiral atoms:
    chi_ats = []
    for atom in from_mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            chi_ats.append(atom)

    # copy over chiral info
    for smi_atom in chi_ats:
        # order of the neighbors around smi_atom
        smi_order = [x.GetIdx() for x in smi_atom.GetNeighbors()]
        
        to_atom = to_mol.GetAtomWithIdx(match[smi_atom.GetIdx()])
        to_atom.SetChiralTag(smi_atom.GetChiralTag())
        
        # check if we need to change that due to neighbor ordering differences
        # current order of neighbors in the to_molol
        mb_order = [x.GetIdx() for x in to_atom.GetNeighbors()]
        
        # with the smiles order:
        mb_smi_order = [match[x] for x in smi_order]
        
        # check if it's a cyclic permutation:
        tmp1 = ','.join([str(x) for x in mb_smi_order])
        tmp2 = ','.join([str(x) for x in smi_order + smi_order])
        
        
        if tmp1 not in tmp2:
            # not cyclic:
            to_atom.InvertChirality()
    
    return to_mol

def mol_from_smiles_pdb(smiles, pdb_filename, infer_names = True, stereo_from_smiles = True):
    """Create rdkit mol from smiles string and pdb file, where the correct connectivity and bond orders are taken from the former and the atom names, residue numbers are taken from the latter.
    It does not care how many pdb structures in the file.

    Parameters
    ----------
    smiles : str
        SMILES representation of molecule.
    pdb_filename : str
        PDB file path to molecule.
    infer_names : bool, optional
        Whether the hydrogen atom names are by inferring based on the heavy atoms the hydrogens are connected to, or taken from the PDB file, by default True.  
    stereo_from_smiles : bool, optional
        Whether to use stereo info from the SMILES, by default True, else the PDB needs to contain a valid 3D conformation from which the stereo info are inferred.

    Returns
    -------
    RDKit mol
        

    Raises
    ------
    ValueError
        If matching cannot be established between the SMILES and pdb.
    """
    ref = Chem.MolFromSmiles(smiles, sanitize = True)

    if infer_names:
        mol = Chem.MolFromPDBFile(pdb_filename, removeHs = True) 
        try:
            out_mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
            
        except Exception as e:
            raise ValueError("Matching to pdb without hydrogens failed: {}".format(e))

        copy_stereo(ref, out_mol) if stereo_from_smiles else AllChem.AssignStereochemistryFrom3D(out_mol) #needs to copy stereo before potentially reordering atoms

        out_mol = reorder_atoms(assign_hydrogen_pdbinfo_blind(Chem.AddHs(out_mol, addCoords = True)))

        return out_mol

    
    else:
        mol = Chem.MolFromPDBFile(pdb_filename, removeHs = False) 
        hydrogen_dict = hydrogen_dict_from_pdbmol(mol)
        try:
            out_mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
        except Exception as e:
            if len(hydrogen_dict) == 0:
                raise ValueError("Matching SMILES to pdb failed: {}".format(e))
            logging.warning("SMILES matching to pdb with hydrogen records failed. Matching to pdb without hydrogens...") 
        
        mol = Chem.MolFromPDBFile(pdb_filename, removeHs = True) 
        try:
            out_mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
        except Exception as e:
            raise ValueError("Matching to pdb without hydrogens failed: {}".format(e))

        copy_stereo(ref, out_mol) if stereo_from_smiles else AllChem.AssignStereochemistryFrom3D(out_mol) #needs to copy stereo before potentially reordering atoms

        if len(hydrogen_dict) == 0:
            out_mol = reorder_atoms(assign_hydrogen_pdbinfo_blind(Chem.AddHs(out_mol, addCoords = True)))
        else:
            try:
                out_mol = reorder_atoms(assign_hydrogen_pdbinfo(Chem.AddHs(out_mol, addCoords = True), hydrogen_dict))
            except Exception as e:
                logging.warning("\n {}. Reside to infering hydrogen names from their attached heavy atom names \n.".format(e))
                out_mol = reorder_atoms(assign_hydrogen_pdbinfo_blind(Chem.AddHs(out_mol, addCoords = True)))
        return out_mol

#TODO maybe better to place somewhere else?
def mol_from_multiple_pdb_files(file_list, removeHs = False): 
    """Create a mol from multiple PDB files, i.e. having multiple conformations.
    Assumes all pdb belong to the same molecule entity

    Parameters
    ----------
    file_list : list
        List of file paths to the PDBs. 
    removeHs : bool, optional
        Have the hydrogens in the molecuel or not, by default False.

    Returns
    -------
    RDKit Mol   
    """
    pdb_string = reduce(lambda a, b : a + b, [open(i, "r").read()[:-4] for i in file_list]) + "END\n"
    return Chem.MolFromPDBBlock(pdb_string, removeHs = removeHs)

def get_atom_mapping(mol, smirks):
    """Get all atom indices that match to substructural pattern.

    Parameters
    ----------
    mol : RDKit Mol
        
    smirks : str
        Substructural pattern for matching.

    Returns
    -------
    list of list
        List of matches, where each match is a list of atom indices.
    """
    qmol = Chem.MolFromSmarts(smirks)
    ind_map = {}
    for atom in qmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            ind_map[map_num - 1] = atom.GetIdx()
    map_list = [ind_map[x] for x in sorted(ind_map)]
    matches = list()
    for match in mol.GetSubstructMatches(qmol, uniquify = False) :
        mas = [match[x] for x in map_list]
        matches.append(tuple(mas))
    return matches

def get_carbonyl_O(mol):
    """Get all carbonyl oxygen indices in a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        
    Returns
    -------
    list of list
        List of matches, where each match is a list of atom indices.
    """
    return [i[0] for i in get_atom_mapping(mol, "[C]=[O:1]")]
def get_amine_H(mol):
    """Get all amine hydrogen indices in a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        
    Returns
    -------
    list of list
        List of matches, where each match is a list of atom indices.
    """
    return [i[0] for i in get_atom_mapping(mol, "[N]-[H:1]")]
def get_1_4_pairs(mol, smirks_1_4 = "[O:1]=[C:2]@;-[NX3:3]-[CX4H3:4]" ):
    """Get all amide 1-4 atom indices in a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        
    Returns
    -------
    list of list
        List of matches, where each match is a list of atom indices.
    """
    return get_atom_mapping(mol, smirks_1_4)
def get_1_5_pairs(mol, smirks_1_5 = "[O:1]=[C]@;-[CX4H1,CX4H2]-[NX3H1]-[H:5]"):
    """Get all amide 1-5 atom indcies indices in a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        
    Returns
    -------
    list of list
        List of matches, where each match is a list of atom indices.
    """
    return get_atom_mapping(mol, smirks_1_5)


def _decide_indices_order(indices):
    """
    arrange indices such the first entry in list has smallest index, the second has the second smallest index
    """

    indices = list(np.roll(indices, -np.argwhere(indices == np.min(indices))[0][0]))
    second_entry, last_entry = indices[1], indices[-1]
    if second_entry > last_entry : #reverse list
        indices = indices[1:] + [indices[0]]
        indices.reverse()
    return [int(i) for i in indices]


def _get_overlap(s1, s2):
    """use for detecting overlap between two sequence of indices 
    the indices are comma-separated numbers in a string

    """
    s = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    overlap = s1[pos_a:pos_a+size] 
    #partial index overlap can occur, limit to only overlap starting and ending with comma
    overlap = overlap[overlap.find(","):]
    overlap = overlap[:overlap.rfind(",")]
    return overlap



# def get_rings(mol, accept_num_fused_atoms = 4):#TODO currently only pairwise ring merge, need to recursively check for ring merges
#     """
#     #shared (fused) part is always the smaller part of the ring, this way the whole ring needs to be at least 9 atoms
#     """
#     # for ring in mol.GetRingInfo().AtomRings():
#     #     yield ring
#     pre_ring_list_len = 0
#     rings = [ring for ring in mol.GetRingInfo().AtomRings()]
#     while len(rings) > pre_ring_list_len:
#         tmp = [",".join([str(i) for i in list(r) * 3]) for r in rings[pre_ring_list_len:]]
#         tmp_inverted = [",".join([str(i) for i in (list(r) * 3)[::-1]]) for r in rings[pre_ring_list_len:]]
        
#         pre_ring_list_len = len(rings)
#         for i in range(len(tmp) - 1):
#             for j in range(i+1, len(tmp)):
#                 overlap = _get_overlap(tmp[i], tmp[j]).strip(",").split(",")
#                 if len(overlap) > 0: #means there is at least some overlap, check inverting a string                
#                     overlap2 = _get_overlap(tmp_inverted[i], tmp[j]).strip(",").split(",") 
#                     if len(overlap) < len(overlap2):
#                         overlap = overlap2
#                         tmp_inverted[i], tmp[i] = tmp[i], tmp_inverted[i] #now i and j indices are in the same direction
#                 if len(overlap) >= accept_num_fused_atoms:
#                     part_a = tmp[j].split(",".join(overlap))[1] #split yields four sequences, one of the two middle pieces is always the complete subsequence
#                     part_b = tmp_inverted[i].split(",".join(overlap[1:-1][::-1]))[1]
#                     rings.append(tuple([int(i) for i in part_a.strip(",").split(",")] + [int(i) for i in part_b.strip(",").split(",")]))
#     return rings
def get_rings(mol, accept_num_fused_atoms = 4): #FIXME currently only allow one fused???
    """Get all (atom indices of) rings in a molecule

    Parameters
    ----------
    mol : RDKit mol
        
    accept_num_fused_atoms : int, optional
        The smallest number of bridging atoms needed to allow two fused rings to be considered as one bigger ring, 
        by default 4, as shared (fused) part is always the smaller part of the ring, this way the whole ring needs to be at least 9 atoms (the size of ring to be considered as macrocycle in our paper)

    Returns
    -------
    list of list
        A list with each member being a list of indices indicating a set of ring indices.
    """
    # for ring in mol.GetRingInfo().AtomRings():
    #     yield ring
    rings = [ring for ring in mol.GetRingInfo().AtomRings()]
    tmp = [",".join([str(i) for i in list(r) * 3]) for r in rings]
    tmp_inverted = [",".join([str(i) for i in (list(r) * 3)[::-1]]) for r in rings]
    for i in range(len(tmp) - 1):
        for j in range(i+1, len(tmp)):
            overlap = _get_overlap(tmp[i], tmp[j]).strip(",").split(",")
            if len(overlap) > 0: #means there is at least some overlap, check inverting a string                
                overlap2 = _get_overlap(tmp_inverted[i], tmp[j]).strip(",").split(",") 
                if len(overlap) < len(overlap2):
                    overlap = overlap2
                    tmp_inverted[i], tmp[i] = tmp[i], tmp_inverted[i] #now i and j indices are in the same direction

            if len(overlap) >= accept_num_fused_atoms:
                part_a = tmp[j].split(",".join(overlap))[1] #split yields four sequences, one of the two middle pieces is always the complete subsequence
                part_b = tmp_inverted[i].split(",".join(overlap[1:-1][::-1]))[1]
                rings.append(tuple([int(i) for i in part_a.strip(",").split(",")] + [int(i) for i in part_b.strip(",").split(",")]))
    return rings

def get_largest_ring(mol):
    """Out put the largest possible ring atom indices of a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        

    Returns
    -------
    list
        List of atom indices of the largest ring.
    """
    out = []
    for r in get_rings(mol):
        if len(r) > len(out):
            out = r
    out = list(out)
    return _decide_indices_order(out)

def get_neighbor_indices(mol, indices):
    """Get the indices of neighbour atoms of selected query atom (query by indices).

    Parameters
    ----------
    mol : RDKit Mol
        
    indices : iterable of int
        The set of query atom indices.

    Returns
    -------
    List
        Neighbour indices of all query atoms.
    """
    out = []
    for i in indices:
        out += [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()]
    out = indices + out
    #set does not necessarily preserve order
    return list(set(out))

def get_neighbour_indices(mol, indices):
    """Get the indices of neighbour atoms of selected query atom (query by indices).

    Parameters
    ----------
    mol : RDKit Mol
        
    indices : iterable of int
        The set of query atom indices.

    Returns
    -------
    List
        Neighbour indices of all query atoms.
    """
    return get_neighbor_indices(mol, indices)


def mol_with_atom_index(mol):
    """Put the atom index of each atom into a property named `molAtomMapNumber`

    Parameters
    ----------
    mol : RDKit Mol
        

    Returns
    -------
    RDKit Mol
        The inputted mol with the aditional `molAtomMapNumber` property.
    """
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def draw_mol_with_property( mol, property, **kwargs):
    """Draw molecule along with atomic properties next to selected atom. Adapted from http://rdkit.blogspot.com/2015/02/new-drawing-code.html

    Parameters
    ----------
    mol : RDKit Mol
        Mol to draw.
    property : dict
        key atom idx, val the property (need to be stringfiable)
    """
    import copy
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem

    def run_from_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False


    AllChem.Compute2DCoords(mol)
    mol = copy.deepcopy(mol) #FIXME do I really need deepcopy?

    for idx in property:
        # opts.atomLabels[idx] =
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', "({})".format( str(property[idx])))

    mol = Draw.PrepareMolForDrawing(mol, kekulize=False) #enable adding stereochem

    if run_from_ipython():
        from IPython.display import SVG, display
        if "width" in kwargs and type(kwargs["width"]) is int and "height" in kwargs and type(kwargs["height"]) is int:
            drawer = Draw.MolDraw2DSVG(kwargs["width"], kwargs["height"])
        else:
            drawer = Draw.MolDraw2DSVG(500,250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        display(SVG(drawer.GetDrawingText().replace("svg:", "")))
    else:
        if "width" in kwargs and type(kwargs["width"]) is int and "height" in kwargs and type(kwargs["height"]) is int:
            drawer = Draw.MolDraw2DCairo(kwargs["width"], kwargs["height"])
        else:
            drawer = Draw.MolDraw2DCairo(500,250) #cairo requires anaconda rdkit
        # opts = drawer.drawOptions()
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        #
        # with open("/home/shuwang/sandbox/tmp.png","wb") as f:
        #     f.write(drawer.GetDrawingText())

        import io
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        buff = io.BytesIO()
        buff.write(drawer.GetDrawingText())
        buff.seek(0)
        plt.figure()
        i = mpimg.imread(buff)
        plt.imshow(i)
        plt.show()
        # display(SVG(drawer.GetDrawingText()))