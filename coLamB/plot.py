# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def display_3d_matrix(data: NDArray, addAxis: bool = True, show: bool = True) -> None:
    """
    Displays a 3D visualization of a 3D matrix using surface plots on the boundaries.

    Args:
        data (NDArray): A 3D numpy array representing the data to be visualized.
        addAxis (bool, optional): Whether to add a colorbar axis to the plot. Default is True.
        show (bool, optional): Whether to display the plot immediately. Default is True.

    Returns:
        None
    """
    ndata = (data - np.min(data)) / (np.max(data) - np.min(data))
    nx, ny, nz = data.shape

    Yx, Zx = np.mgrid[0 : (ny + 1), 0 : (nz + 1)]
    Xx = np.zeros(Yx.shape) + nx

    Zy, Xy = np.mgrid[0 : (nz + 1), 0 : (nx + 1)]
    Yy = np.zeros(Zy.shape)

    Xz, Yz = np.mgrid[0 : (nx + 1), 0 : (ny + 1)]
    Zz = np.zeros(Xz.shape) + nz

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, nx + 1)
    ax.set_ylim(-1, ny)
    ax.set_zlim(0, nz + 1)
    col = plt.cm.BrBG(ndata[-1, :, :])
    surf = ax.plot_surface(
        Xx,
        Yx,
        Zx,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.BrBG(ndata[-1, :, :]),
        shade=False,
    )
    surf = ax.plot_surface(
        Xy,
        Yy,
        Zy,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.BrBG(ndata[:, 0, :].T),
        shade=False,
    )
    surf = ax.plot_surface(
        Xz,
        Yz,
        Zz,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.BrBG(ndata[:, :, -1]),
        shade=False,
        cmap="BrBG",
    )
    if addAxis:
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_ticks([0, 1.0])
        cbar.set_ticklabels([np.min(data), np.max(data)])
        plt.cm.BrBG(ndata[-1, :, :])
    if show:
        plt.show()


if __name__ == "__main__":
    data = np.linspace(-5, 10, num=3 * 4 * 5).reshape((3, 4, 5))
    display_3d_matrix(data)
