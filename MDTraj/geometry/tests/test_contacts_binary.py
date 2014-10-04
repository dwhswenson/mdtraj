#
# Authors: David W.H. Swenson
#

import os
import numpy as np
from mdtraj.testing import get_fn, eq, skipif
import itertools
from mdtraj import geometry
import mdtraj as md

# TODO: early testing -- eventually this should not be needed
import sys
sys.path.append("..")
from contacts_binary import CellDecomposition,neighbor_atoms,neighbor_residues


# TESTS TO RUN
# 1. without box vectors, we should get fake box vectors without periodicity
# 2. with traj, we should get results for each frame
# 3. example where the protein is separated from water by a periodic image
#    (also meaning that your box numbers might be negative

def test_contacts_binary_0():
    #traj = md.load( get_fn('1am7_corrected.xtc'), 
    #                top=get_fn('1am7_protein.pdb') )
    #traj = md.load( get_fn('1vii_sustiva_water.pdb') )
    traj = md.load( get_fn('alanine-dipeptide-explicit.pdb') )

    frame = traj[0]
    #a = np.array([1,0,0])
    #b = np.array([1,1,0])
    #c = np.array([0,0,1])
    #frame.unitcell_vectors = np.array([ [ a, b, c] ])
    #cell0 = CellDecomposition(frame, 0.3)

    waters = [r.index for r in traj.top.residues if r.name == "HOH"]
    protein = [r.index for r in traj.top.chain(0).residues 
                            if r.name != "HOH" ]
    n_dist, n_pairs = neighbor_residues(traj[0], [protein,waters])
    c_dist, c_pairs = md.compute_contacts(
                                traj,
                                list(itertools.product(protein, waters)))

    cps = [cp for (cp, cd) in zip(c_pairs, c_dist[0]) if cd<0.6]
    cds = [cd for (cp, cd) in zip(c_pairs, c_dist[0]) if cd<0.6]
    print (len(n_pairs[0]), len(cps))

    

if __name__ == "__main__":
    test_contacts_binary_0()
