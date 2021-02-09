import json

import numpy as np
import pytest
from simtk import unit

from openff.toolkit.tests.utils import (
    get_context_potential_energy,
    get_data_file_path,
    requires_rdkit,
)
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField


@requires_rdkit
@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize(
    "mol",
    [
        "ethanol.sdf",
        "methane_multiconformer.sdf",
        "CID20742535_anion.sdf",
    ],
)
def test_reference(constrained, mol):
    """Minimal regression test comparing molecule energies to energies computed
    by version 0.8.0 of the toolkit"""
    # TODO: Also test periodic vs. vacuum
    with open(get_data_file_path("reference_energies/reference_0.8.0.json"), "r") as fi:
        reference = json.loads(fi.read())

    name = mol + "_"
    if not constrained:
        name += "un"
    name += "constrained"
    reference_energy = reference[name]

    omm_sys, positions, off_top = _build_system(
        mol=mol,
        constrained=constrained,
    )

    simulation = _build_simulation(omm_sys=omm_sys, off_top=off_top)
    derived_energy = _get_energy(simulation=simulation, positions=positions)

    try:
        np.testing.assert_almost_equal(
            actual=derived_energy / unit.kilojoule_per_mole,
            desired=reference_energy,
            decimal=5,
        )
    except AssertionError as e:
        msg = (
            str(e)
            + "\nAll forces:\n\t"
            + str(_get_energy_by_force_group(simulation, positions))
        )
        raise AssertionError(msg)


def generate_reference():
    """Function to generate reference files, if this script is called directly by a Python interpreter.
    This is ignored by pytest while running tests."""
    reference = dict()
    for mol in [
        "ethanol.sdf",
        "methane_multiconformer.sdf",
        "CID20742535_anion.sdf",
    ]:
        for constrained in [True, False]:
            omm_sys, positions, off_top = _build_system(
                mol=mol,
                constrained=constrained,
            )

            simulation = _build_simulation(omm_sys=omm_sys, off_top=off_top)
            energy = _get_energy(simulation=simulation, positions=positions)

            name = mol + "_"
            if not constrained:
                name += "un"
            name += "constrained"
            reference.update({name: energy / unit.kilojoule_per_mole})

    import openff.toolkit

    toolkit_version = openff.toolkit.__version__

    with open(f"reference_{toolkit_version}.json", "w") as json_out:
        json.dump(reference, json_out)


def _build_system(mol, constrained):
    if constrained:
        parsley = ForceField("openff-1.0.0.offxml")
    else:
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_file(get_data_file_path("molecules/" + mol), file_format="sdf")

    if type(mol) == Molecule:
        off_top = mol.to_topology()
        positions = mol.conformers[0]
    elif type(mol) == list:
        # methane_multiconformer case is a list of two mols
        off_top = Topology()
        for mol_i in mol:
            off_top.add_molecule(mol_i)
        positions = (
            np.vstack([mol[0].conformers[0], mol[1].conformers[0]]) * unit.angstrom
        )

    from openff.toolkit.utils.toolkits import (
        AmberToolsToolkitWrapper,
        RDKitToolkitWrapper,
        ToolkitRegistry,
    )

    toolkit_registry = ToolkitRegistry(
        toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
    )

    omm_sys = parsley.create_openmm_system(off_top, toolkit_registry=toolkit_registry)

    return omm_sys, positions, off_top


def _build_simulation(omm_sys, off_top):
    """Given an OpenMM System, initialize a barebones OpenMM Simulation."""
    from simtk import openmm

    # Use OpenMM to compute initial and minimized energy for all conformers
    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    omm_top = off_top.to_openmm()
    simulation = openmm.app.Simulation(omm_top, omm_sys, integrator, platform)

    return simulation


def _get_energy(simulation, positions):
    """Given an OpenMM simulation and position, return its energy"""
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy()

    return energy


def _get_energy_by_force_group(simulation, positions):
    simulation.context.setPositions(positions)

    force_names = {f.__class__.__name__ for f in simulation.system.getForces()}
    group_to_force = {i: force_name for i, force_name in enumerate(force_names)}
    force_to_group = {force_name: i for i, force_name in group_to_force.items()}

    for force in simulation.system.getForces():
        force.setForceGroup(force_to_group[force.__class__.__name__])

    energies = get_context_potential_energy(
        context=simulation.context,
        positions=positions,
        by_force_group=True,
    )

    return {group_to_force[idx]: energies[idx] for idx in energies.keys()}


if __name__ == "__main__":
    generate_reference()
