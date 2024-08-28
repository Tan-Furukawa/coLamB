import numpy as np
import tqdm
from numpy.typing import NDArray


def get_vogit_from_pair(i: int, j: int) -> int:
    """
    Maps a pair of indices (i, j) to the corresponding Voigt notation index.

    Args:
        i (int): The first index (0, 1, or 2).
        j (int): The second index (0, 1, or 2).

    Returns:
        int: The corresponding Voigt notation index (0 to 5).

    Raises:
        ValueError: If the indices do not correspond to a valid Voigt notation.
    """
    if i == 0 and j == 0:
        return 0
    elif i == 1 and j == 1:
        return 1
    elif i == 2 and j == 2:
        return 2
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        return 3
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        return 4
    elif (i == 0 and j == 1) or (i == 1 and j == 0):
        return 5
    else:
        return ValueError("invalid i or j")


def get_Vogit_pair(i: int, j: int, k: int, l: int) -> tuple[int, int]:
    """
    Converts a pair of index pairs (i, j) and (k, l) to their corresponding Voigt notation indices.

    Args:
        i (int): The first index of the first pair.
        j (int): The second index of the first pair.
        k (int): The first index of the second pair.
        l (int): The second index of the second pair.

    Returns:
        tuple[int, int]: A tuple of two integers representing the Voigt notation indices.
    """
    I = get_vogit_from_pair(i, j)
    J = get_vogit_from_pair(k, l)
    return I, J


def green_tensor(
    kx: NDArray, ky: NDArray, kz: NDArray, cp: NDArray, cm: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Computes the Green's tensor and the intermediate omega components for a given set of wave vectors and elastic constants.

    Args:
        kx (NDArray): The wave vector components in the x direction.
        ky (NDArray): The wave vector components in the y direction.
        kz (NDArray): The wave vector components in the z direction.
        cp (NDArray): The elastic stiffness tensor of the precipitate phase.
        cm (NDArray): The elastic stiffness tensor of the matrix phase.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing:
            - tmatx: The Green's tensor in Fourier space.
            - omeg11: The first component of the omega tensor used in the Green's tensor calculation.
    """

    Nx = len(kx)
    Ny = len(ky)
    Nz = len(kz)

    c = 0.5 * (cm + cp)

    omeg11 = np.zeros((Nx, Ny, Nz))
    omeg12 = np.zeros((Nx, Ny, Nz))
    omeg13 = np.zeros((Nx, Ny, Nz))
    omeg22 = np.zeros((Nx, Ny, Nz))
    omeg23 = np.zeros((Nx, Ny, Nz))
    omeg33 = np.zeros((Nx, Ny, Nz))

    for i in range(Nx):
        for j in range(Ny):
            for l in range(Nz):

                a = np.zeros((3, 3))
                k = [kx[i], ky[j], kz[l]]

                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            for s in range(3):
                                vpair = get_Vogit_pair(p, r, q, s)
                                a[p, q] += c[vpair[0], vpair[1]] * k[r] * k[s]

                a11 = a[0, 0]
                a12 = a[0, 1]
                a13 = a[0, 2]
                a22 = a[1, 1]
                a23 = a[1, 2]
                a33 = a[2, 2]

                det = (
                    a11 * a22 * a33
                    - a11 * a23 * a23
                    - a12 * a12 * a33
                    + a12 * a23 * a13
                    + a13 * a12 * a23
                    - a13 * a22 * a13
                )

                if det == 0:
                    omeg11[i, j, l] = 0
                    omeg12[i, j, l] = 0
                    omeg13[i, j, l] = 0
                    omeg22[i, j, l] = 0
                    omeg23[i, j, l] = 0
                    omeg33[i, j, l] = 0
                else:  # omeg_ik = (C_ijkl * k_j * k_l)^-1
                    omeg11[i, j, l] = (a22 * a33 - a23**2) / det
                    omeg12[i, j, l] = (-a12 * a33 + a13 * a23) / det
                    omeg13[i, j, l] = (a12 * a23 - a13 * a22) / det
                    omeg22[i, j, l] = (a11 * a33 - a13**2) / det
                    omeg23[i, j, l] = (-a11 * a23 + a12 * a13) / det
                    omeg33[i, j, l] = (a11 * a22 - a12**2) / det

    tmatx = np.zeros((Nx, Ny, Nz, 3, 3, 3, 3))

    # gmatx: Greens tensor
    # 1/2 * (k_k * G_pl + k_l * G_pk) * k_q
    for i in tqdm.tqdm(range(Nx)):
        for j in range(Ny):
            for l in range(Ny):
                gmatx = np.zeros((3, 3))
                gmatx[0, 0] = omeg11[i, j, l]
                gmatx[0, 1] = omeg12[i, j, l]
                gmatx[0, 2] = omeg13[i, j, l]
                gmatx[1, 0] = omeg12[i, j, l]
                gmatx[1, 1] = omeg22[i, j, l]
                gmatx[1, 2] = omeg23[i, j, l]
                gmatx[2, 0] = omeg13[i, j, l]
                gmatx[2, 1] = omeg23[i, j, l]
                gmatx[2, 2] = omeg33[i, j, l]

                # position vector
                dvect = np.zeros(3)
                dvect[0] = kx[i]
                dvect[1] = ky[j]
                dvect[2] = kz[l]

                # Green operator
                for kk in range(3):
                    for ll in range(3):
                        for ii in range(3):
                            for jj in range(3):
                                tmatx[i, j, l, kk, ll, ii, jj] = 0.25 * (
                                    gmatx[ll, ii] * dvect[jj] * dvect[kk]
                                    + gmatx[kk, ii] * dvect[jj] * dvect[ll]
                                    + gmatx[ll, jj] * dvect[ii] * dvect[kk]
                                    + gmatx[kk, jj] * dvect[ii] * dvect[ll]
                                )
    return tmatx, omeg11
