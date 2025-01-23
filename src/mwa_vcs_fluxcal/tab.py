#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import logging
import numpy as np
from mwalib import MetafitsContext, Pol
from mwa_hyperbeam import FEEBeam as PrimaryBeam
from astropy.constants import c as sol

import mwa_vcs_fluxcal

# Define MWA location
MWA_CENTRE_LON = 116.67081524 * u.deg
MWA_CENTRE_LAT = -26.70331940 * u.deg
MWA_CENTRE_H = 377.8269 * u.m
MWA_CENTRE_CABLE_LEN = 0.0 * u.m

MWA_LOCATION = EarthLocation.from_geodetic(
    lon=MWA_CENTRE_LON, lat=MWA_CENTRE_LAT, height=MWA_CENTRE_H
)

logger = mwa_vcs_fluxcal.get_logger()

def getPrimaryBeamPower(
    metadata: MetafitsContext,
    freq_hz: float,
    alt: float | np.ndarray,
    az: float | np.ndarray,
    stokes: str = "I",
    zenithNorm: bool = True,
    show_path: bool = False,
) -> dict:
    """Calculate the primary beam response (full Stokes) for a
    given observation over a grid of the sky.

    :param metadata: A mwalib.MetafitsContext object that contains the
                     array configuration and delay settings.
    :type metadata: MetafitsContext
    :param freq_hz: Observing radio frequency, in Hz.
    :type freq_hz: float
    :param alt: Desired altitude for the pointing direction, in radians.
                Can be an array.
    :type alt: np.ndarray, float
    :param az: Desired azimuth for the pointing direction, in radians.
               Can be an array.
    :type az: np.ndarray, float
    :param stokes: Which Stokes parameters to compute and return.
                   A string containing some unique combination of "IQUV".
                   Values are returned in the order requested here.
    :type stokes: str
    :param zenithNorm: Whether to normalise the primary beam response to
                       the value at zenith (maximum sensitivity).
                       Defaults to True.
    :type zenithNorm: bool, optional
    :param show_path: Show the einsum optimization path. Defaults to False.
    :type show_path: bool, optional
    :return: The primary beam response for each requested Stokes parameter
             over the provided sky positions. Note: The position axis is
             flattened and needs to be reshaped based on the input az/alt
             arguments (or however the user desires).
    :rtype: dict[str, np.ndarray]
    """
    za = np.pi / 2 - alt
    beam = PrimaryBeam()

    logger.info("Calculating Jones matrices")
    jones = beam.calc_jones_array(
        np.array([az]).flatten(),
        np.array([za]).flatten(),
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        zenithNorm,
    )
    logger.info(f"Creating sky power Stokes {stokes} response")
    J = jones.reshape(-1, 2, 2)  # shape = (npix, 2, 2)
    K = np.conjugate(J).T  # = J^H, shape = (2, 2, npix)

    # For the coherency matrix products transformed by the Jones matrices, we
    # can use the Pauli spin matrices and simple matrix operations to extract
    # the final Stokes parameters. Effectively using the formalism of the
    # "polarisation measurement equation" of Hamaker (2000) and van Straten (2004).
    rho = dict(
        sI=np.matrix([[1, 0], [0, 1]]),  # sigma0, provides I
        sU=np.matrix([[0, 1], [1, 0]]),  # sigma1, provides U
        sV=np.matrix([[0, -1j], [1j, 0]]),  # sigma2, provides V
        sQ=np.matrix([[1, 0], [0, -1]]),  # sigma3, provides Q
    )
    # Multiplying the above spin matrices on the left by the Jones matrix,
    # and on the right by the Hermitian transpose of the Jones matrix will
    # retrieve the Stokes response of the instrument (modulo a scaling factor).
    # i.e., for each of the N sky positions sampled,
    #
    #   Tr[ J @ S0 @ K ] = 2I
    #   Tr[ J @ S1 @ K ] = 2U
    #   Tr[ J @ S2 @ K ] = -i(U - iV) + i(U + iV) = -2V
    #   Tr[ J @ S3 @ K ] = 2Q
    #
    # where Tr is the trace operator, @ implies matrix multiplication,
    # and "i" is the imaginary unit.

    # Here, we figure out the optimal contraction path once, and then just use
    # that for each Stokes parameter. (There is possibly a more efficient combination
    # of operations might scale better, but this is still rapid.)
    einsum_path = np.einsum_path("Nki,ij,jkN->N", J, rho["sI"], K, optimize="optimal")
    if show_path:
        logger.debug(einsum_path[0])
        logger.debug(einsum_path[1])
    # This einsum does the following operations:
    # - N is our "batch" dimension, so we can do a batch of N matrix multiplications
    # - first, we do the multiplication of N (k x i) Jones matrices onto our (i x j) Pauli matrix
    # - then we do the multiplication of the N (j x k) composite matrices onto the inverse Jones matrix (j x k)
    # - finally, the "->N" symbol implies the trace (sum of diagonals) of each N matrices

    stokes_response = dict()
    for st in stokes:
        # From the Stokes parameter letter, retrieve the correct spin matrix
        rho_mat = rho[f"s{st}"]

        # Determine the scale factor required to apply after matrix operations.
        # Here we use casefold() to ensure comparison is case-agnostic
        if st.casefold() in "IQU".casefold():
            scale = 1 / 2
        elif st.casefold() == "V".casefold():
            scale = -1 / 2
        else:
            logger.critical(f"Unrecognized Stokes parameter: st={st}!")
            raise ValueError(f"Unrecognized Stokes parameter: st={st}!")

        stokes_response.update(
            {
                f"{st}": scale
                * np.einsum(
                    "Nki,ij,jkN->N", J, rho_mat, K, optimize=einsum_path[0]
                ).real
                # We explicitly take the real part here due to floating-point
                # precision leaving some very small imaginary components in the result
            }
        )

    return stokes_response


def extractWorkingTilePositions(metadata: MetafitsContext) -> np.ndarray:
    """Extract tile position information required for beamforming and/or
    computing the array factor quantity from a metafits structure.
    Flagged tiles are automatically excluded from the result.

    :param metadata: An MWALIB MetafitsContext structure
                     containing the array layout information.
    :type metadata: MetafitsContext
    :return: Working tile positions and electrical lengths for
             beamforming. Formatted as an array of arrays, where
             each item in the outer array is:
                [east_m, north_m, height_m, electrical_length_m]
             for a single tile.
    :rtype: np.ndarray
    """
    # Gather the tile positions into a "vector" for each tile
    tile_positions = np.array(
        [
            np.array(
                [
                    rf.east_m,
                    rf.north_m,
                    rf.height_m,
                    rf.electrical_length_m - MWA_CENTRE_CABLE_LEN.value,
                ]
            )
            for rf in metadata.rf_inputs
            if rf.pol == Pol.X
        ]
    )

    # Gather the flagged tile information from the metafits information
    # and remove those tiles from the above vector
    tile_flags = np.array([rf.flagged for rf in metadata.rf_inputs if rf.pol == Pol.X])
    tile_positions = np.delete(tile_positions, np.where(tile_flags == True), axis=0)

    return tile_positions


def calcGeometricDelays(
    positions: np.ndarray, freq_hz: float, alt: float, az: float
) -> np.ndarray:
    """Compute the geometric delay phases for each element position in order to
    "phase up" to the provided position at a specific frequency. These are the
    phasors used in a beamforming operation.

    :param positions: An array or element position vectors, including their
                      equivalent electrical length, in metres.
    :type positions: np.ndarray
    :param freq_hz: Observing radio frequency, in Hz.
    :type freq_hz: float
    :param alt: Desired altitude for the pointing direction, in radians.
                Can be an array.
    :type alt: np.ndarray, float
    :param az: Desired azimuth for the pointing direction, in radians.
               Can be an array.
    :type az: np.ndarray, float
    :return: The required phasors needed to rotate the element patterns to
             each requested az/alt pair.
    :rtype: np.ndarray
    """
    # Create the unit vector(s)
    u = np.array(
        [
            np.cos(alt) * np.sin(az),  # unit E
            np.cos(alt) * np.cos(az),  # unit N
            np.sin(alt),  # unit H
            -np.ones_like(alt),  # cable length (-ve as it is subtracted)
        ]
    )

    # Compute the equivalent delay length for each tile
    # (Use tensor dot product so we can choose to keep the
    # dimensionality of the alt/az grid and continue using
    # broadcasting rules efficiently.)
    w = np.tensordot(positions, u, axes=1)
    # From the numpy.tensordot documentation:
    #    The third argument can be a single non-negative integer_like scalar, N;
    #    if it is such, then the last N dimensions of a and the first N dimensions
    #    of b are summed over.

    # Convert to a time delay
    dt = w / sol.value

    # Construct the phasor
    phase = 2 * np.pi * freq_hz * dt
    phasor = np.exp(1.0j * phase)

    return phasor


def calcArrayFactorPower(look_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    """Compute the array factor power from a given pointing phasor
    and one or more target directions.

    :param look_w: The complex phasor representing the tile phases
        in the desired "look direction".
    :type look_w: np.array, complex
    :param target_w: The complex phasor(s) representing the tile
        phases required to look in the desired sample directions.
    :type target_w: np.ndarray, complex
    :return: The absolute array factor power, for each given
        target direction.
    :rtype: np.ndarray
    """
    # At this stage, the shape of target_w = (nant, n_ra, n_dec) and while the shape of look_w = (nant,)
    logger.info("Summing over antennas")
    sum_over_antennas = np.tensordot(np.conjugate(look_w), target_w, axes=1)
    # From the numpy.tensordot documentation:
    #    The third argument can be a single non-negative integer_like scalar, N;
    #    if it is such, then the last N dimensions of a and the first N dimensions
    #    of b are summed over.

    # The array factor power is normalised to the number of elements
    # included in the sum (i.e., length of the `look_w` vector).
    logger.info("Averaging over array and converting to power")
    afp = (np.absolute(sum_over_antennas) / look_w.size) ** 2

    return afp
