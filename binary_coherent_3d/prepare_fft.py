# %%
import cupy as cp
import numpy as np


class CDArray: ...


def prepare_fft(
    Nx: int,
    Ny: int,
    Nz: int,
    dx: float,
    dy: float,
    dz: float,
    eta_x: float = 1,
    eta_y: float = 1,
    eta_z: float = 1,
) -> tuple[
    CDArray,
    CDArray,
    CDArray,
    CDArray,
    CDArray,
    CDArray,
    CDArray,
    CDArray,
]:
    """
    Prepares the necessary arrays for performing a Fourier transform (FFT) in 3D space, including wave vectors and their associated magnitudes.

    Args:
        Nx (int): Number of grid points in the x direction.
        Ny (int): Number of grid points in the y direction.
        Nz (int): Number of grid points in the z direction.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        dz (float): Grid spacing in the z direction.
        eta_x (float, optional): Scaling factor for the x direction wave vector. Default is 1.
        eta_y (float, optional): Scaling factor for the y direction wave vector. Default is 1.
        eta_z (float, optional): Scaling factor for the z direction wave vector. Default is 1.

    Returns:
        tuple[CDArray, CDArray, CDArray, CDArray, CDArray, CDArray, CDArray, CDArray]:
            - kx (CDArray): The wave vector in the x direction.
            - ky (CDArray): The wave vector in the y direction.
            - kz (CDArray): The wave vector in the z direction.
            - k2 (CDArray): The squared magnitude of the wave vectors.
            - k4 (CDArray): The fourth power of the magnitude of the wave vectors.
            - kx_mat (CDArray): The x component of the wave vector matrix.
            - ky_mat (CDArray): The y component of the wave vector matrix.
            - kz_mat (CDArray): The z component of the wave vector matrix.
    """
    Nx21 = Nx // 2 + 1
    Ny21 = Ny // 2 + 1
    Nz21 = Nz // 2 + 1

    Nx2 = Nx + 2
    Ny2 = Ny + 2
    Nz2 = Nz + 2
    pi = 3.141592

    delkx = (2.0 * pi) / (Nx * dx)
    delky = (2.0 * pi) / (Ny * dy)
    delkz = (2.0 * pi) / (Nz * dz)

    kx = np.zeros(Nx2)
    ky = np.zeros(Ny2)
    kz = np.zeros(Nz2)

    for i in range(1, Nx21 + 1):
        fk1 = (i - 1) * delkx
        kx[i - 1] = fk1
        kx[Nx2 - i - 1] = -fk1

    for j in range(1, Ny21 + 1):
        fk2 = (j - 1) * delky
        ky[j - 1] = fk2
        ky[Ny2 - j - 1] = -fk2

    for k in range(1, Nz21 + 1):
        fk3 = (k - 1) * delkz
        kz[k - 1] = fk3
        kz[Nz2 - k - 1] = -fk3

    kx = kx[:Nx]
    ky = ky[:Ny]
    kz = kz[:Nz]

    k2 = np.zeros((Nx, Ny, Nz))
    k2_any = np.zeros((Nx, Ny, Nz))
    k4 = np.zeros((Nx, Ny, Nz))
    k4_any = np.zeros((Nx, Ny, Nz))

    kx_mat = np.zeros((Nx, Ny, Nz))
    ky_mat = np.zeros((Nx, Ny, Nz))
    kz_mat = np.zeros((Nx, Ny, Nz))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                kx_mat[i, j, k] = kx[i]
                ky_mat[i, j, k] = ky[j]
                kz_mat[i, j, k] = kz[k]
                k2[i, j, k] = kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2
                k2_any[i, j, k] = (
                    eta_x * kx[i] ** 2 + eta_y * ky[j] ** 2 + eta_z * kz[k] ** 2
                )
                k4[i, j, k] = k2[i, j, k] ** 2
                k4_any[i, j, k] = k2_any[i, j, k]

    return (
        cp.asarray(kx),
        cp.asarray(ky),
        cp.asarray(kz),
        cp.asarray(k2),
        cp.asarray(k4),
        cp.asarray(kx_mat),
        cp.asarray(ky_mat),
        cp.asarray(kz_mat),
    )


if __name__ == "__main__":
    Nx = 4
    Ny = 4
    Nz = 4
    dx = 1.0
    dy = 1.0
    dz = 1.0
    kx, ky, kz, k2, k4, kx_mat, ky_mat, kz_mat = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    print(kx)

    Nx = 8
    Ny = 8
    Nz = 8
    dx = 1.0
    dy = 1.0
    dz = 1.0
    kx, ky, kz, k2, k4, kx_mat, ky_mat, kz_mat = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    print(kx)

# %%
