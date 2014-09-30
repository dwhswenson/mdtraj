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
            pass

        self.hinv = np.linalg.inv(boxvectors)
        self.hx = 1 # TODO: this is only for cubic

        # split the box vectors into subcells
        self.ncells = []
        self.boxheights = []
        for d in range(3):
            # TODO: check that the box height uses correct shape (doesn't
            # matter for cubic boxes, but will matter in general
            self.boxheights.append(1.0/np.linalg.norm(self.hinv[d,:]))
            self.ncells.append(int(self.boxheights[d] / self.maxdist))
        self.cellvectors = boxvectors / self.ncells
        self.cell_abc = [np.linalg.norm(self.cellvectors[d,:]) 
                            for d in range(3)]
                            


    def assign(self, grouplabel, group):
        """
        Assigns (or reassigns) a group of atoms to the appropriate cells. 
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
            # convert xyz coordinates to abc coordinates
            abc = self.hx*pos
            # assign that to a box
            boxnum = []
            for d in range(3):
                num = abc[d] / self.cell_abc[d] // 1
                num = cellwrap(num, self.ncells[d])
                boxnum.append(num)
            # Use the tuple of the box number as a dictionary key for the
            # box. This is convenient for python tricks, and also means that
            # we only store grid boxes that we actually use.
            if tuple(boxnum) in cells.keys():
                cells[tuple(boxnum)].append(atom_i)
            else:
                cells[tuple(boxnum)] = [atom_i]

        self.cells_atoms[grouplabel] = cells


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


def compute_contacts_binary(traj, groups, scheme='closest-heavy',
                            maxdist=None, 
                            celldecomp=None, output='residues'):
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
    residue_pairs :

    Examples
    --------


    See Also
    --------
    mdtraj.geometry.compute_contacts : Computes all contacts in the
        `contacts` list
    """
    if traj.topology is None:
        raise ValueError('Binary contacts requires a topology')



    scheme_atoms = { 'ca':'ca', 'closest-heavy':'heavy', 'closest':'all'}

    # default maxdist for different atom types
    if maxdist==None:
        maxdist = { 
                    'ca' : 1.0, 'closest-heavy' : 0.6, 'closest':0.3
                  }[scheme]

    for frame in traj:
        if not celldecomp:
            celldecomp = CellDecomposition(frame,maxdist,scheme_atoms[scheme])
        if celldecomp.types != scheme_atoms[scheme]:
            pass # raise error? warning?
        # TODO: come up with a way to speed up assign based on previous
        # frame, probably something Verlet list-based
        celldecomp.assign('group0', groups[0])
        celldecomp.assign('group1', groups[1])
        cells0 = celldecomp.cells_atoms['group0'].keys()
        cells1 = celldecomp.cells_atoms['group1'].keys()
        # loop over the smaller group; sometimes my codes just hypnotize me
        if len(cells0) > len(cells1):
            biggie = cells0
            smalls = cells1
            sizeorder = 1
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
        # circumstances; e.g., if we want a simple yes/no on residue-residue
        # interactions ... the above might be a separate fcn

        # make the atom-atom pairs from the relevant cell pairs
        distance_pairs = []
        for pair in cellpairs:
            distance_pairs.extend( itertools.product(
                    celldecomp.cells_atoms['group0'][pair[0]],
                    celldecomp.cells_atoms['group1'][pair[1]] ) )
        
