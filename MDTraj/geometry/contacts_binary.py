#
# Authors: David W.H. Swenson
#
# Closely modeled after compute_contacts function.
#

import numpy as np
import math
from mdtraj.utils import ensure_type
from mdtraj.geometry import _geometry
from mdtraj.core import element
import mdtraj as md
import itertools

class CellDecomposition(object):
    """ Divides periodic box into cells such that all points within
    `maxdist` of a point in a given cell is either in that cell or one of
    its 26 neighbors.

    Attributes
    ----------
    cellvectors : np.ndarray, shape=(3,3)
        vectors for a single subcell of the unit cell
    types : { 'ca', 'heavy', 'all' }
        type of atoms to include in cell list:
        'ca' : alpha carbons
        'heavy' : non-hydrogen atoms
        'all' : all atoms
    maxdist : float
        maximum distance of interest: all points on a sphere with this
        radius, centered on a given point, are in the neighbor cells of the
        given point
    """
    def __init__(self, frame, maxdist, types='heavy'):
        """ Initialize the object, and set up the cell list cells. """
        self.cellvectors = []
        self.cells_atoms = {}
        self.cells_residues = {}
        self.types = types
        self.maxdist = maxdist
        self.frame = frame
        
        # figure out what the cells should look like: take the box vectors
        # and the maxdist and figure out the shape of the box we're
        # splitting up
        # NOTE: geometry.distance() gives the user the option to ignore
        # periodicity; I haven't done that here, although it would be an
        # easy addition
        if frame._have_unitcell:
            boxvectors = frame.unitcell_vectors[0]
        else:
            # create fake box vectors based on min/max x,y,z; create an
            # empty shell around the whole thing (by adding 1 to the number
            # of subcells in each direction) so we ignore periodicity
            # TODO
            pass

        self.hinv = np.linalg.inv(boxvectors)
        hx = [] 
        hinvx = []

        # split the box vectors into subcells
        self.ncells = []
        self.boxheights = []
        for d in range(3):
            # TODO: check that the box height uses correct shape (doesn't
            # matter for cubic boxes, but will matter in general
            hx.append(boxvectors[d,:]/np.linalg.norm(boxvectors[d,:]))
            hinvx.append(self.hinv[d,:]/np.linalg.norm(self.hinv[d,:]))
            self.boxheights.append(1.0/np.linalg.norm(self.hinv[d,:]))
            self.ncells.append(int(self.boxheights[d] / self.maxdist))
        self.cellvectors = boxvectors / self.ncells
        self.cell_abc = [np.linalg.norm(self.cellvectors[d,:]) 
                            for d in range(3)]
        self.hx = np.array(hx).T
        self.hinvx = np.array(hinvx).T
        #print boxvectors
        #print self.ncells
        #print self.cellvectors
        #print self.hx
        #print self.hinvx



    def assign(self, grouplabel, group):
        """
        Assigns (or reassigns) atoms from a group of residue indices to the
        appropriate cells. 
        """
        cells = {}
        xyz = ensure_type(self.frame.xyz, dtype=np.float32, ndim=3,
                            name='traj.xyz', shape=(1,None,3))
        group_atoms = []
        topol = self.frame.top
        for res in group:
            if self.types == 'ca':
                group_atoms.extend([a.index for a in topol.residue(res).atoms
                                    if a.name.lower() == 'ca'])
            elif self.types == 'heavy':
                group_atoms.extend([a.index for a in topol.residue(res).atoms
                                    if not (a.element == element.hydrogen)])
            elif self.types == 'all':
                group_atoms.extend([a.index for a in topol.residue(res).atoms])

        # take the xyz of each atom and assign it to a cell
        for atom_i in group_atoms:
            pos = xyz[0][atom_i]
            boxnum = self.cell_for_position(pos)
            # Use the tuple of the box number as a dictionary key for the
            # box. This is convenient for python tricks, and also means that
            # we only store grid boxes that we actually use.
            if tuple(boxnum) in cells.keys():
                cells[tuple(boxnum)].append(atom_i)
            else:
                cells[tuple(boxnum)] = [atom_i]

        self.cells_atoms[grouplabel] = cells
 
    def cell_for_position(self, pos):
        # convert xyz coordinates to abc coordinates
        #abc = np.dot(self.hx,pos)
        abc = np.dot(self.hinvx,pos)
        # assign that to a box
        boxnum = []
        for d in range(3):
            num = abc[d] / self.cell_abc[d] // 1
            num = cellwrap(num, self.ncells[d])
            boxnum.append(num)
        return tuple(boxnum)



    def neighborhood(self,cellid):
        ''' Gives the cells in the "neighborhood" of the given cell.
        Parameters
        ----------
        cellid : 3-tuple
            central cell for this neighborhood

        Returns
        -------
        neighbors : list of 3-tuple
            The 27 cell boxes, including `cellid`, which define the
            neighborhood. All points within `maxdist` of any point in the
            cell `cellid` are in one of these 27 cells. Includes periodic
            wrapping.
        '''
        neighbors = []
        for deltax in -1,0,1:
            t0 = cellwrap(cellid[0]+deltax, self.ncells[0])
            for deltay in -1,0,1:
                t1 = cellwrap(cellid[1]+deltay, self.ncells[1])
                for deltaz in -1,0,1:
                    t2 = cellwrap(cellid[2]+deltaz, self.ncells[2])
                    neighbors.append( tuple([t0,t1,t2]) )
        return neighbors

        
def cellwrap(cellid, ncells):
    '''Convenience method for PBCs'''
    while cellid >= ncells:
        cellid -= ncells
    while cellid < 0:
        cellid += ncells
    return cellid

def neighbor_atoms_frame(frame, groups, maxdist=None, scheme='closest-heavy'):
    # TODO: raise error if groups is of length!=2
    scheme_atoms = { 'ca':'ca', 'closest-heavy':'heavy', 'closest':'all'}
    # default maxdist for different atom types
    if maxdist==None:
        maxdist = { 
                    'ca' : 1.0, 'closest-heavy' : 0.6, 'closest':0.3
                  }[scheme]

    celldecomp = CellDecomposition(frame,maxdist,scheme_atoms[scheme])

    if celldecomp.types != scheme_atoms[scheme]:
        pass # raise error? warning?
    celldecomp.assign('group0', groups[0])
    celldecomp.assign('group1', groups[1])
    cells0 = celldecomp.cells_atoms['group0'].keys()
    cells1 = celldecomp.cells_atoms['group1'].keys()
    # set the loop to go over the smaller group (autopick query/haystack)
    # [sometimes my codes just hypnotize me]
    if len(cells0) > len(cells1):
        biggie = cells0
        smalls = cells1
        sizeorder = 1
        # sizeorder lets us match the groups back to the right order
    else:
        biggie = cells1
        smalls = cells0
        sizeorder = -1
    cellpairs = []
    for cell in smalls:
        neighborhood = celldecomp.neighborhood(cell)
        # check if the biggie is chillin' in smalls' hood; add cells to
        # list if so
        for neighb in neighborhood:
            if neighb in biggie:
                if sizeorder==1:
                    cellpairs.append([neighb, cell])
                elif sizeorder==-1:
                    cellpairs.append([cell,neighb])
                else:
                    raise ValueError('WTF? sizeorder is wonky')
    
    # TODO: from here we might change things to speed it up in various
    # circumstances. In particular, if we just want a simple yes/no on
    # residue contact we skip atom pairs if their residues have already
    # be id'd as in contact

    # make the atom-atom pairs from the relevant cell pairs
    distance_pairs = []
    #print cellpairs
    for pair in cellpairs:
        # option to reduce memory usage: send distance_pairs off to be
        # calculated if a certain number of pairs are generated; then
        # reset distance_pairs

        # TODO: except in particular circumstances, we'll want to prevent
        # pairs from the same residue (possibly neighboring residues) from
        # being added, as well. But may be faster to just calc all the
        # distances and filter those after selecting the ones that are close
        # enough to matter (fewer branches).
        distance_pairs.extend( itertools.product(
                celldecomp.cells_atoms['group0'][pair[0]],
                celldecomp.cells_atoms['group1'][pair[1]] ) )


    atom_distances = md.compute_distances(frame, distance_pairs)[0]

    for (p,d) in zip(distance_pairs, atom_distances):
        pos0, pos1 = (frame.xyz[0,p[0],:], frame.xyz[0,p[1],:])
        #print "neighbors:", p,d
        #print "  ", pos0, celldecomp.cell_for_position(pos0)
        #print "  ", pos1, celldecomp.cell_for_position(pos1)

    testpair = (257, 1678)
    pos0 = frame.xyz[0,testpair[0],:]
    pos1 = frame.xyz[0,testpair[1],:]
    #print "Test for pair", testpair, md.compute_distances(frame, [testpair])
    #print "   ", pos0, celldecomp.cell_for_position(pos0)
    #print "   ", pos1, celldecomp.cell_for_position(pos1)
    #print "       ", np.dot(celldecomp.hx.T,pos1)

    n0=0
    for cell in cells0:
        n0+=len(celldecomp.cells_atoms['group0'][cell])
    n1=0
    for cell in cells1:
        n1+=len(celldecomp.cells_atoms['group1'][cell])
    #print n0,n1

    # TODO: this zipping/unzipping is probably slower/more memory
    # intensive than direct iteration
    zipped=[(pair,dist) for pair,dist in zip(distance_pairs, atom_distances) 
                        if dist < maxdist]
    #print "Distances:", len(distance_pairs), "/",
    #print len(groups[0])*len(groups[1]), len(zipped)
    if len(zipped) > 0:
        f_atom_pairs, f_distances = zip(*zipped)
    else:
        f_atom_pairs = []
        f_distances = []

    return f_distances, f_atom_pairs


def neighbor_atoms(traj, groups, maxdist=None, scheme='closest-heavy'):
    """Compute distances for contacts only if distance is closer than a
    given value.

    This is particularly useful if trying to identify important contacts in
    a simulation, e.g., all contacts between two chains. We decompose this
    system into cells of size `maxdist` (rounded up) so that we only need to
    look at particles in neighboring cells. As such, it scales linearly with
    the number of atoms, not quadratically.

    Parameters
    ----------
    traj : 
    groups : 
    scheme :
    maxdist : float
        cutoff distance; if not given, value depends on the `scheme` chosen
    celldecomp : 
        if we've already generated a cell decomp, we can use it here
    output : 'residues' or 'atoms'

    Returns
    -------
    distances :
    atom_pairs : 

    Examples
    --------


    See Also
    --------
    mdtraj.geometry.compute_contacts : Computes all contacts in the
        `contacts` list
    """
    if traj.topology is None:
        raise ValueError('Binary contacts requires a topology')

    atom_pairs = []
    distances = []
    nframe=0
    for frame in traj:
        nframe += 1
        print "Starting frame", nframe
        f_distances,f_atom_pairs = neighbor_atoms_frame(frame, groups,
                                                        scheme, maxdist)
        atom_pairs.append(f_atom_pairs)
        distances.append(f_distances)

    return distances, atom_pairs

def filter_atompairs_to_residuepairs(top, atom_pairs, values, order='min'):
    sgn = { 'min' : 1 , 'max' : -1 }[order]
    res_dict = {}
    for (pair, val) in zip(atom_pairs, values):
        respair = ( top.atom(pair[0]).residue.index, 
                    top.atom(pair[1]).residue.index )
        if not (respair in res_dict and res_dict[respair] < sgn*val):
            res_dict[respair] = val
    return zip(*[(v,r) for r,v in res_dict.iteritems()])

def filter_duplicate_residues(res_pairs, values, partner=0, order='min'):
    sgn = { 'min' : 1 , 'max' : -1 }[order]
    res_dict = {}
    for (pair, val) in zip(res_pairs, values):
        myres = pair[partner]
        if not (myres in res_dict and res_dict[myres][0] < sgn*val):
            res_dict[myres] = (val, pair)
    return zip(*res_dict.values())

def neighbor_residues(traj, groups, maxdist=None, scheme='closest-heavy'):
    """ Determines minimum distance """
    res_dists = []
    res_pairs = []
    #atomgroup0 = []
    #atomgroup1 = []
    #for r in groups[0]:
    #    atomgroup0.extend( [a.index for a in traj.top.residue(r).atoms] )
    #for r in groups[1]:
    #    atomgroup1.extend( [a.index for a in traj.top.residue(r).atoms] )
    fnum=0
    for frame in traj:
        fnum += 1
        #print "Starting frame", fnum
        atom_dists, atom_pairs = neighbor_atoms_frame(frame, 
                                                    groups,
                                                    maxdist, scheme)
        if atom_dists!=[]:
            f_res_dists, f_res_pairs = filter_atompairs_to_residuepairs(
                                                traj.top, atom_pairs, 
                                                atom_dists)
        else:
            f_res_dists = []
            f_res_pairs = []
        res_dists.append(f_res_dists)
        res_pairs.append(f_res_pairs)
    return res_dists, res_pairs

    # this next line would have provided above with nearest member from
    # group B to each member of group A (e.g, nearest water to each residue)
    #final_dists, final_pairs = filter_duplicate_residues(res_pairs,res_dists)

