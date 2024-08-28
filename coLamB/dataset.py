# %%
import numpy as np


class DataSet:
    """
    A class to represent the material properties and interdiffusion function for a specific dataset.

    Attributes:
        stiffness (numpy.ndarray): The stiffness tensor from Haussuehl 1993, represented as a 6x6 matrix.
        ei0 (numpy.ndarray): The numpy.ndarray of chemical strain tensor estimated from Kroll et al., 1986. ([e11, e22, e33, e23, e13, e12])
        interdiffusion_fn (function): A polynomial function representing the interdiffusion coefficient, estimated from Foland 1974 (Or) and Kasper 1975 (Ab).
    """

    def __init__(self) -> None:
        # Stiffness tensor of Haussuehl 1993
        self.stiffness = np.array(
            [
                [68.72390598, 49.18591017, 38.535481, 0.0, -2.48400375, 0.0],
                [49.18591017, 176.8, 15.41408983, 0.0, 1.14718693, 0.0],
                [38.535481, 15.41408983, 134.70513202, 0.0, -29.67158601, 0.0],
                [0.0, 0.0, 0.0, 13.51828739, 0.0, -1.05426758],
                [-2.48400375, 1.14718693, -29.67158601, 0.0, 30.535481, 0.0],
                [0.0, 0.0, 0.0, -1.05426758, 0.0, 39.28171261],
            ]
        )

        # chemical strain estimated from Kroll et al., 1986
        self.ei0 = np.array([0.05400176, 0.01152419, 0.01306945, 0.0, 0.01530671, 0.0])

        # interdiffusion function estimated from Foland 1974 (Or) and Kasper 1975 (Ab)
        self.interdiffusion_fn = (
            lambda x: 3.76071663e-02
            + 1.25668600e01 * x
            + -5.48393485e01 * x**2
            + 9.57901774e01 * x**3
            + -7.74194773e01 * x**4
            + 2.39347525e01 * x**5
        )
