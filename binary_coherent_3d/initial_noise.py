import cupy as np
from numpy.typing import NDArray


def add_initial_noise(Nx: int, Ny: int, Nz: int, c0: float, noise: float, seed:int=123)->NDArray:
    """
    Generates an initial concentration field with added random noise.

    Args:
        Nx (int): The number of grid points in the x direction.
        Ny (int): The number of grid points in the y direction.
        Nz (int): The number of grid points in the z direction.
        c0 (float): The initial average concentration value.
        noise (float): The amplitude of the noise to be added to the concentration field.
        seed (int, optional): The seed for the random number generator to ensure reproducibility. Default is 123.

    Returns:
        NDArray: A 3D numpy array of shape (Nx, Ny, Nz) representing the concentration field with added noise.
    """

    con = np.zeros((Nx, Ny, Nz))

    rng = np.random.default_rng(seed=seed)
    con = c0 + noise * (0.5 - rng.random((Nx, Ny, Nz)))
    # np.random.seed(123)
    # con = c0 + noise * (0.5 - np.random.rand(Nx, Ny))

    return con
