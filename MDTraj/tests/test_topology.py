##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2014 Stanford University and the Authors
#
# Authors: Kyle A. Beauchamp
# Contributors: Robert McGibbon, Matthew Harrigan
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

import os
import tempfile
import mdtraj as md
import numpy as np
from mdtraj.utils.six.moves import cPickle
from mdtraj.utils import import_
from mdtraj.testing import (get_fn, eq, DocStringFormatTester, skipif,
                            assert_raises)

try:
    from simtk.openmm import app
    HAVE_OPENMM = True
except ImportError:
    HAVE_OPENMM = False

try:
    from simtk.unit import amu
    HAVE_SIMTKUNIT = True
except ImportError:
    HAVE_SIMTKUNIT = False

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


@skipif(not HAVE_OPENMM)
def test_topology_openmm():
    topology = md.load(get_fn('1bpi.pdb')).topology

    # the openmm trajectory doesn't have the distinction
    # between resSeq and index, so if they're out of whack
    # in the openmm version, that cant be preserved
    for residue in topology.residues:
        residue.resSeq = residue.index
    mm = topology.to_openmm()
    assert isinstance(mm, app.Topology)
    topology2 = md.Topology.from_openmm(mm)
    eq(topology, topology2)

@skipif(not HAVE_OPENMM)
def test_topology_to_openmm_arbitrary_atoms():
    topology = md.Topology()
    elemA = md.element.Element(1, "LJ_A", "A", 1.0)
    elemX = md.element.Element(2, "LJ_X", "X", 2.0)
    c1 = topology.add_chain()
    c2 = topology.add_chain()
    # chain 1 consists of 2 residues, AAX-AXA
    # chain 2 consists of 3 residues, AXA-XXAX-AA
    r1 = topology.add_residue("AAX", c1, 1)
    r2 = topology.add_residue("AXA", c1, 2)
    r3 = topology.add_residue("AXA", c2, 3)
    r4 = topology.add_residue("XXAX", c2, 4)
    r5 = topology.add_residue("AA", c2, 5)
    num = 1
    for res in [r1, r2, r3, r4, r5]:
        for a in res.name:
            element = md.element.Element.getBySymbol(a)
            topology.add_atom(a+str(num), element, res)
            num += 1
    openmm_topology = topology.to_openmm()
    mdtraj_topology = topology.from_openmm(openmm_topology)
    eq(topology, mdtraj_topology)

@skipif(not (HAVE_OPENMM and HAVE_SIMTKUNIT))
def test_topology_from_openmm_arbitrary_atoms():
    # same as above, except we start out with the openmm topology
    topology = app.Topology()
    elemA = app.element.Element(1, "LJ_A", "A", 1.0 * amu)
    elemX = app.element.Element(2, "LJ_X", "X", 2.0 * amu)
    c1 = topology.addChain()
    c2 = topology.addChain()
    r1 = topology.addResidue("AAX", c1)
    r2 = topology.addResidue("AXA", c1)
    r3 = topology.addResidue("AXA", c2)
    r4 = topology.addResidue("XXAX", c2)
    r5 = topology.addResidue("AA", c2)
    num = 1
    for res in [r1, r2, r3, r4, r5]:
        for a in res.name:
            element = app.element.Element.getBySymbol(a)
            topology.addAtom(a+str(num), element, res)
            num += 1
    mdtraj_topology = md.Topology.from_openmm(topology)
    openmm_topology = mdtraj_topology.to_openmm()
    eq(topology, openmm_topology)


@skipif(not HAVE_OPENMM)
def test_topology_openmm_boxes():
    u = import_('simtk.unit')
    traj = md.load(get_fn('1vii_sustiva_water.pdb'))
    mmtop = traj.topology.to_openmm(traj=traj)
    box = mmtop.getUnitCellDimensions() / u.nanometer


@skipif(not HAVE_PANDAS)
def test_topology_pandas():
    topology = md.load(get_fn('native.pdb')).topology
    atoms, bonds = topology.to_dataframe()

    topology2 = md.Topology.from_dataframe(atoms, bonds)
    eq(topology, topology2)


@skipif(not HAVE_PANDAS)
def test_topology_pandas_TIP4PEW():
    topology = md.load(get_fn('GG-tip4pew.pdb')).topology
    atoms, bonds = topology.to_dataframe()

    topology2 = md.Topology.from_dataframe(atoms, bonds)
    eq(topology, topology2)

def test_topology_numbers():
    topology = md.load(get_fn('1bpi.pdb')).topology
    assert len(list(topology.atoms)) == topology.n_atoms
    assert len(list(topology.residues)) == topology.n_residues
    assert all([topology.atom(i).index == i for i in range(topology.n_atoms)])

@skipif(not HAVE_PANDAS)
def test_topology_unique_elements_bpti():
    traj = md.load(get_fn('bpti.pdb'))
    top, bonds = traj.top.to_dataframe()
    atoms = np.unique(["C", "O", "N", "H", "S"])
    eq(atoms, np.unique(top.element.values))

def test_chain():
    top = md.load(get_fn('bpti.pdb')).topology
    chain = top.chain(0)
    assert chain.n_residues == len(list(chain.residues))

    atoms = list(chain.atoms)
    assert chain.n_atoms == len(atoms)
    for i in range(chain.n_atoms):
        assert atoms[i] == chain.atom(i)

def test_residue():
    top = md.load(get_fn('bpti.pdb')).topology
    residue = top.residue(0)
    assert len(list(residue.atoms)) == residue.n_atoms
    atoms = list(residue.atoms)
    for i in range(residue.n_atoms):
        assert residue.atom(i) == atoms[i]

def test_nonconsective_resSeq():
    t = md.load(get_fn('nonconsecutive_resSeq.pdb'))
    yield lambda : eq(np.array([r.resSeq for r in t.top.residues]), np.array([1, 3, 5]))
    df1 = t.top.to_dataframe()
    df2 = md.Topology.from_dataframe(*df1).to_dataframe()
    yield lambda : eq(df1[0], df2[0])

    # round-trip through a PDB load/save loop
    fd, fname = tempfile.mkstemp(suffix='.pdb')
    os.close(fd)
    t.save(fname)
    t2 = md.load(fname)
    yield lambda : eq(df1[0], t2.top.to_dataframe()[0])
    os.unlink(fname)

def test_pickle():
    # test pickling of topology (bug #391)
    cPickle.loads(cPickle.dumps(md.load(get_fn('bpti.pdb')).topology))

def test_atoms_by_name():
    top = md.load(get_fn('bpti.pdb')).topology

    atoms = list(top.atoms)
    for atom1, atom2 in zip(top.atoms_by_name('CA'), top.chain(0).atoms_by_name('CA')):
        assert atom1 == atom2
        assert atom1 in atoms
        assert atom1.name == 'CA'

    assert len(list(top.atoms_by_name('CA'))) == sum(1 for _ in atoms if _.name == 'CA')
    assert top.residue(15).atom('CA') == [a for a in top.residue(15).atoms if a.name == 'CA'][0]

    assert_raises(KeyError, lambda: top.residue(15).atom('sdfsdsdf'))

def test_select_atom_indices():
    top = md.load(get_fn('native.pdb')).topology

    yield lambda: eq(top.select_atom_indices('alpha'), np.array([8]))
    yield lambda: eq(top.select_atom_indices('minimal'),
                     np.array([4, 5, 6, 8, 10, 14, 15, 16, 18]))

    assert_raises(ValueError, lambda: top.select_atom_indices('sdfsdfsdf'))

@skipif(not HAVE_OPENMM)
def test_top_dataframe_openmm_roundtrip():
    t = md.load(get_fn('2EQQ.pdb'))
    top, bonds = t.top.to_dataframe()
    t.topology = md.Topology.from_dataframe(top, bonds)
    omm_top = t.top.to_openmm()


def test_n_bonds():
    t = md.load(get_fn('2EQQ.pdb'))
    for atom in t.top.atoms:
        if atom.element.symbol == 'H':
            assert atom.n_bonds == 1
        elif atom.element.symbol == 'C':
            assert atom.n_bonds in [3, 4]
        elif atom.element.symbol == 'O':
            assert atom.n_bonds in [1, 2]


def test_load_unknown_topology():
    try:
        md.load(get_fn('frame0.dcd'), top=get_fn('frame0.dcd'))
    except IOError as e:
        # we want to make sure there's a nice error message than includes
        # a list of the supported topology formats.
        assert all(s in str(e) for s in ('.pdb', '.psf', '.prmtop'))
    else:
        assert False  # fail

