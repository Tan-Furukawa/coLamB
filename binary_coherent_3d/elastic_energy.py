import cupy as cp
import numpy as np


class CDArray: ...


def get_i_j_index(i: int) -> tuple[int, int]:
    """
    Returns the corresponding (i, j) index tuple based on the input integer.

    Args:
        i (int): An integer value between 0 and 5 inclusive.

    Returns:
        tuple[int, int]: A tuple of two integers representing the (i, j) indices.

    Raises:
        TypeError: If `i` is not in the range 0 to 5.
    """
    if i == 0:
        return (0, 0)
    elif i == 1:
        return (1, 1)
    elif i == 2:
        return (2, 2)
    elif i == 3:
        return (1, 2)
    elif i == 4:
        return (0, 2)
    elif i == 5:
        return (0, 1)
    else:
        raise TypeError("i is 0 to 5")


def solve_elasticity(
    tmatx: CDArray,
    cm: CDArray,
    c_p: CDArray,
    ea: CDArray,
    ei0: CDArray,
    con: CDArray,
    c0: float,
) -> tuple[CDArray, CDArray, CDArray]:
    """
    Solves the elasticity problem for a given material system using iterative methods.

    Args:
        tmatx (CDArray): The Green operator tensor in Fourier space.
        cm (CDArray): The elastic stiffness tensor of the matrix phase.
        c_p (CDArray): The elastic stiffness tensor of the precipitate phase.
        ea (CDArray): The applied strain tensor (not implemented).
        ei0 (CDArray): The chemical strain tensor.**DO NOT use Voigt notation for the chemical strain. For the chemical strain, use a `numpy.ndarray` that specifies the components of the 3x3 tensor `[[ei11, ei12, ei13], [ei12, ei22, ei23], [ei13, ei23, ei33]]` in the order `[ei11, ei22, ei33, ei23, ei13, ei12]`.**
        con (CDArray): The concentration field of the precipitate phase.
        c0 (float): The bulk concentration.

    Returns:
        tuple[CDArray, CDArray, CDArray]: A tuple containing:
            - delsdc0: The derivative of strain energy with respect to concentration.
            - s: The stress tensor field. (s[:,:,:,i] | i in (0..5))
            - el: The elastic strain tensor field. (el[:,:,:,i] | i in (0..5))
    """
    Nx, Ny, Nz = con.shape

    niter = 10
    tolerance = 0.0001
    old_norm = None

    # initialize stress
    s = cp.zeros((Nx, Ny, Nz, 6))

    # initialize total strain
    e = cp.zeros((Nx, Ny, Nz, 6))

    # elastic strain
    el = cp.zeros((Nx, Ny, Nz, 6))

    # --- eigenstrains:
    # ei * (c(x) - c0) where c(x) is molar fraction of precipitants at point x
    ei = cp.zeros((Nx, Ny, Nz, 6))
    for i in range(6):
        ei[:, :, :, i] = ei0[i] * (con - c0)

    # calculate effective elastic constants
    c = cp.zeros((Nx, Ny, Nz, 6, 6))
    ci = c_p - cm
    I, J = ci.shape
    for i in range(I):
        for j in range(J):
            c[:, :, :, i, j] = con * c_p[i, j] + (1 - con) * cm[i, j]

    # -- Green operator:
    # e[k] = e[k] - Γ:s[k]

    smatx = cp.zeros((Nx, Ny, Nz, 3, 3), dtype=cp.complex128)
    ek = cp.zeros((Nx, Ny, Nz))
    for iter in range(niter):

        # --- take the stresses & strains to Fourier space
        smatx[:, :, :, 0, 0] = cp.fft.fftn(s[:, :, :, 0])
        smatx[:, :, :, 1, 1] = cp.fft.fftn(s[:, :, :, 1])
        smatx[:, :, :, 2, 2] = cp.fft.fftn(s[:, :, :, 2])
        smatx[:, :, :, 1, 2] = cp.fft.fftn(s[:, :, :, 3])
        smatx[:, :, :, 2, 1] = smatx[:, :, :, 1, 2]
        smatx[:, :, :, 0, 2] = cp.fft.fftn(s[:, :, :, 4])
        smatx[:, :, :, 2, 0] = smatx[:, :, :, 0, 2]
        smatx[:, :, :, 0, 1] = cp.fft.fftn(s[:, :, :, 5])
        smatx[:, :, :, 1, 0] = smatx[:, :, :, 0, 1]

        for i in range(6):
            I, J = get_i_j_index(i)
            ek = cp.fft.fftn(e[:, :, :, i])
            for kk in range(3):
                for ll in range(3):
                    ek = ek - tmatx[:, :, :, I, J, kk, ll] * smatx[:, :, :, kk, ll]
            e[:, :, :, i] = cp.real(cp.fft.ifftn(ek))

        el = e - ei

        # Calculate stresses:
        # e:  fluctuation strain
        # ei: eigen strain
        # s = C * (ea + e - ei)
        (
            s[:, :, :, 0],
            s[:, :, :, 1],
            s[:, :, :, 2],
            s[:, :, :, 3],
            s[:, :, :, 4],
            s[:, :, :, 5],
        ) = (
            c[:, :, :, 0, 0] * el[:, :, :, 0]
            + c[:, :, :, 0, 1] * el[:, :, :, 1]
            + c[:, :, :, 0, 2] * el[:, :, :, 2]
            + (
                c[:, :, :, 0, 3] * el[:, :, :, 3]
                + c[:, :, :, 0, 4] * el[:, :, :, 4]
                + c[:, :, :, 0, 5] * el[:, :, :, 5]
            )
            * 2,
            c[:, :, :, 0, 1] * el[:, :, :, 0]
            + c[:, :, :, 1, 1] * el[:, :, :, 1]
            + c[:, :, :, 1, 2] * el[:, :, :, 2]
            + (
                c[:, :, :, 1, 3] * el[:, :, :, 3]
                + c[:, :, :, 1, 4] * el[:, :, :, 4]
                + c[:, :, :, 1, 5] * el[:, :, :, 5]
            )
            * 2,
            c[:, :, :, 0, 2] * el[:, :, :, 0]
            + c[:, :, :, 1, 2] * el[:, :, :, 1]
            + c[:, :, :, 2, 2] * el[:, :, :, 2]
            + (
                c[:, :, :, 2, 3] * el[:, :, :, 3]
                + c[:, :, :, 2, 4] * el[:, :, :, 4]
                + c[:, :, :, 2, 5] * el[:, :, :, 5]
            )
            * 2,
            c[:, :, :, 0, 3] * el[:, :, :, 0]
            + c[:, :, :, 1, 3] * el[:, :, :, 1]
            + c[:, :, :, 2, 3] * el[:, :, :, 2]
            + (
                c[:, :, :, 3, 3] * el[:, :, :, 3]
                + c[:, :, :, 3, 4] * el[:, :, :, 4]
                + c[:, :, :, 3, 5] * el[:, :, :, 5]
            )
            * 2,
            c[:, :, :, 0, 4] * el[:, :, :, 0]
            + c[:, :, :, 1, 4] * el[:, :, :, 1]
            + c[:, :, :, 2, 4] * el[:, :, :, 2]
            + (
                c[:, :, :, 3, 4] * el[:, :, :, 3]
                + c[:, :, :, 4, 4] * el[:, :, :, 4]
                + c[:, :, :, 4, 5] * el[:, :, :, 5]
            )
            * 2,
            c[:, :, :, 0, 5] * el[:, :, :, 0]
            + c[:, :, :, 1, 5] * el[:, :, :, 1]
            + c[:, :, :, 2, 5] * el[:, :, :, 2]
            + (
                c[:, :, :, 3, 5] * el[:, :, :, 3]
                + c[:, :, :, 4, 5] * el[:, :, :, 4]
                + c[:, :, :, 5, 5] * el[:, :, :, 5]
            )
            * 2,
        )

        # ---check convergence:
        sum_stres = (
            s[:, :, :, 0]
            + s[:, :, :, 1]
            + s[:, :, :, 2]
            + s[:, :, :, 3]
            + s[:, :, :, 4]
            + s[:, :, :, 5]
        )
        normF = cp.linalg.norm(sum_stres)

        if iter != 0:
            conver = abs((normF - old_norm) / old_norm)
            if conver <= tolerance:
                break
        old_norm = normF

    el = e - ei

    # del E / del c
    # E = Cijkl * et_ij * et_kl
    # C = 1/2 (Cp + Cm) - (1/2 - c)(Cp - Cm), del C /del c = Cp - Cm
    # del E / del c = (Cp - Cm) * et_ij * et_kl - 2ei0 * Cijkl * et_kl * δ_ij
    # del et / del c = ei0

    ci = c_p - cm

    delsdc0 = 0.5 * (
        (
            (
                ci[0, 0] * el[:, :, :, 0]
                + ci[0, 1] * el[:, :, :, 1]
                + ci[0, 2] * el[:, :, :, 2]
                + (
                    ci[0, 3] * el[:, :, :, 3]
                    + ci[0, 4] * el[:, :, :, 4]
                    + ci[0, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 0]
            + (
                ci[0, 1] * el[:, :, :, 0]
                + ci[1, 1] * el[:, :, :, 1]
                + ci[1, 2] * el[:, :, :, 2]
                + (
                    ci[1, 3] * el[:, :, :, 3]
                    + ci[1, 4] * el[:, :, :, 4]
                    + ci[1, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 1]
            + (
                ci[0, 2] * el[:, :, :, 0]
                + ci[1, 2] * el[:, :, :, 1]
                + ci[2, 2] * el[:, :, :, 2]
                + (
                    ci[2, 3] * el[:, :, :, 3]
                    + ci[2, 4] * el[:, :, :, 4]
                    + ci[2, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 2]
            + (
                ci[0, 3] * el[:, :, :, 0]
                + ci[1, 3] * el[:, :, :, 1]
                + ci[2, 3] * el[:, :, :, 2]
                + (
                    ci[3, 3] * el[:, :, :, 3]
                    + ci[3, 4] * el[:, :, :, 4]
                    + ci[3, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 3]
            * 2
            + (
                ci[0, 4] * el[:, :, :, 0]
                + ci[1, 4] * el[:, :, :, 1]
                + ci[2, 4] * el[:, :, :, 2]
                + (
                    ci[3, 4] * el[:, :, :, 3]
                    + ci[4, 4] * el[:, :, :, 4]
                    + ci[4, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 4]
            * 2
            + (
                ci[0, 5] * el[:, :, :, 0]
                + ci[1, 5] * el[:, :, :, 1]
                + ci[2, 5] * el[:, :, :, 2]
                + (
                    ci[3, 5] * el[:, :, :, 3]
                    + ci[4, 5] * el[:, :, :, 4]
                    + ci[5, 5] * el[:, :, :, 5]
                )
                * 2
            )
            * el[:, :, :, 5]
            * 2
        )
        - 2.0
        * (
            (
                c[:, :, :, 0, 0] * ei0[0]
                + c[:, :, :, 0, 1] * ei0[1]
                + c[:, :, :, 0, 2] * ei0[2]
                + (
                    c[:, :, :, 0, 3] * ei0[3]
                    + c[:, :, :, 0, 4] * ei0[4]
                    + c[:, :, :, 0, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 0]
            + (
                c[:, :, :, 0, 1] * ei0[0]
                + c[:, :, :, 1, 1] * ei0[1]
                + c[:, :, :, 1, 2] * ei0[2]
                + (
                    c[:, :, :, 1, 3] * ei0[3]
                    + c[:, :, :, 1, 4] * ei0[4]
                    + c[:, :, :, 1, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 1]
            + (
                c[:, :, :, 0, 2] * ei0[0]
                + c[:, :, :, 1, 2] * ei0[1]
                + c[:, :, :, 2, 2] * ei0[2]
                + (
                    c[:, :, :, 2, 3] * ei0[3]
                    + c[:, :, :, 2, 4] * ei0[4]
                    + c[:, :, :, 2, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 2]
            + (
                c[:, :, :, 0, 3] * ei0[0]
                + c[:, :, :, 1, 3] * ei0[1]
                + c[:, :, :, 2, 3] * ei0[2]
                + (
                    c[:, :, :, 3, 3] * ei0[3]
                    + c[:, :, :, 3, 4] * ei0[4]
                    + c[:, :, :, 3, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 3]
            * 2
            + (
                c[:, :, :, 0, 4] * ei0[0]
                + c[:, :, :, 1, 4] * ei0[1]
                + c[:, :, :, 2, 4] * ei0[2]
                + (
                    c[:, :, :, 3, 4] * ei0[3]
                    + c[:, :, :, 4, 4] * ei0[4]
                    + c[:, :, :, 4, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 4]
            * 2
            + (
                c[:, :, :, 0, 5] * ei0[0]
                + c[:, :, :, 1, 5] * ei0[1]
                + c[:, :, :, 2, 5] * ei0[2]
                + (
                    c[:, :, :, 3, 5] * ei0[3]
                    + c[:, :, :, 4, 5] * ei0[4]
                    + c[:, :, :, 5, 5] * ei0[5]
                )
                * 2
            )
            * el[:, :, :, 5]
            * 2
        )
    )
    return delsdc0, s, el


# %%
