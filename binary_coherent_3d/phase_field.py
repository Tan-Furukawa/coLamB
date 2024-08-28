# %%
import pickle
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from binary_coherent_3d.initial_noise import add_initial_noise
from binary_coherent_3d.prepare_fft import prepare_fft
from binary_coherent_3d.green_tensor import green_tensor
from binary_coherent_3d.elastic_energy import solve_elasticity
from binary_coherent_3d.free_energy import get_free_energy
from binary_coherent_3d import plot as mplt
from binary_coherent_3d import save as msave

from typing import Literal, Union
from numpy.typing import NDArray


class CoherentBinary3D(object):
    """
    A class to perform phase field simulations for a binary coherent system with elasticity.

    Attributes:
        method (str): The method used for phase field calculation, either "linear" or "nonlinear".
        save_path (str): The path where results will be saved.
        cm_input (CDArray): The stiffness tensor of the matrix phase.
        cp_input (CDArray): The stiffness tensor of the precipitate phase.
        ei0 (CDArray): The chemical strain tensor.**DO NOT use Voigt notation for the chemical strain. For the chemical strain, use a `numpy.ndarray` that specifies the components of the 3x3 tensor `[[ei11, ei12, ei13], [ei12, ei22, ei23], [ei13, ei23, ei33]]` in the order `[ei11, ei22, ei33, ei23, ei13, ei12]`.**
        dirname (str): The name of the directory for saving results.
        con (CDArray): The concentration field.

    Methods:
        __init__(save_path, stiffness, chemical_strain, method): Initializes the phase field simulation.
        set_all(): Prepares all parameters and initial conditions for the simulation.
        exec(): Executes the phase field simulation.
        set_initial_parameters(): Sets the initial parameters for the simulation.
        make_calculation_parameters(): Calculates derived parameters required for the simulation.
        make_save_file(): Creates the necessary directories for saving results.
        save_instance(): Saves the current state of the instance to a file.
        load_instance(full_dir_path=None): Loads a previously saved instance from a file.
        prepare_result_variables(): Prepares variables for storing the results of the simulation.
        is_included_target_file_in_directory(directory, target_name): Checks if a target file is in the specified directory.
        calculate_green_tensor(): Calculates or loads the Green's tensor used in elasticity calculations.
        calculate_phase_filed(method): Performs the phase field calculation.
    """

    def __init__(
        self,
        save_path: str,
        stiffness: NDArray,
        chemical_strain: NDArray,
        method: Literal["linear", "nonlinear"] = "linear",
    ) -> None:
        """
        Initializes the CoherentBinary3D instance with the specified parameters.

        Args:
            save_path (str): The directory path where the simulation results will be saved.
            stiffness (NDArray): The stiffness tensor of the material.
            chemical_strain (NDArray): The chemical strain tensor.**DO NOT use Voigt notation for the chemical strain. For the chemical strain, use a `numpy.ndarray` that specifies the components of the 3x3 tensor `[[ei11, ei12, ei13], [ei12, ei22, ei23], [ei13, ei23, ei33]]` in the order `[ei11, ei22, ei33, ei23, ei13, ei12]`.**
            method (str, optional): The method for phase field calculation, either "linear" or "nonlinear". Default is "linear".

        Returns:
            None
        """
        self.method = method
        self.save_path = save_path
        self.set_initial_parameters()
        self.cm_input = cp.asarray(stiffness)
        self.cp_input = cp.asarray(stiffness)
        # ei0_5 should half size.
        self.ei0 = cp.asarray(chemical_strain)

    def set_all(self) -> None:
        """
        Sets up all parameters and variables necessary for the simulation, including directory creation and parameter calculation.

        Returns:
            None
        """
        self.dirname = msave.make_dir_name()
        self.make_save_file()
        self.save_instance()
        self.make_calculation_parameters()
        self.prepare_result_variables()
        self.calculate_green_tensor()

    def exec(self) -> None:
        """
        Executes the entire phase field simulation process.

        Returns:
            None
        """
        self.set_all()
        self.calculate_phase_filed(method=self.method)

    def set_initial_parameters(self) -> None:
        """
        Sets the initial parameters for the phase field simulation.

        Returns:
            None
        """
        self.noise_per_step = 0
        self.iter = 0
        self.roop_start_from = 1
        self.Nx = 64
        self.Ny = 64
        self.Nz = 64
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.nstep = 3000
        self.nsave = 1000
        self.nprint = 100
        self.dtime = 5.0e-2
        self.coefA = 1.0
        self.c0 = 0.4
        self.mobility = 1.0
        self.grad_coef = 2.0
        self.noise = 0.01
        self.__R = 8.31446262
        self.P = 1 * 10**5  # [Pa]
        self.T = 700  # [K]
        self.accel = 1

        self.interdiffusion_fn = lambda x: x * (1 - x)
        self.w_or_input = lambda T, P: (22820 - 6.3 * T + 0.461 * P / 10**5)
        self.w_ab_input = lambda T, P: (19550 - 10.5 * T + 0.327 * P / 10**5)
        self.v_or = 8.60 * 13.2 * 7.18 * np.sin(116 / 180 * np.pi)  # [A^3]
        self.v_ab = 8.15 * 12.85 * 7.12 * np.sin(116 / 180 * np.pi)  # [A^3]
        # Cij[GPa] * 10^9 * v[Å] * 10*(-30) * NA[/mol] = [/mol]

        # applied strains
        self.ea = cp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # eigen strain (del e / del c)

        self.n_unit_volume = 4

    def make_calculation_parameters(self) -> None:
        """
        Calculates the parameters required for the simulation, such as elastic constants and wave vectors.

        Returns:
            None
        """
        self.__wor = self.w_or_input(self.T, self.P) / (self.__R * self.T)
        self.__wab = self.w_ab_input(self.T, self.P) / (self.__R * self.T)

        N_A = 6.02 * 10.0**23
        # mol per unit volume (m^3) = (n in unit volume) /(unit vol) * 1 / N_A
        n_or = 1 / (self.v_or * 10 ** (-30)) * self.n_unit_volume / N_A
        n_ab = 1 / (self.v_ab * 10 ** (-30)) * self.n_unit_volume / N_A
        self.__n0 = n_or * self.c0 + (1 - self.c0) * n_ab

        # cm* = cm / (nRT)
        self.__cm = self.cm_input * 10.0**9 / (self.__R * self.T) / self.__n0
        self.__cp = self.cp_input * 10.0**9 / (self.__R * self.T) / self.__n0

        (
            self.__kx,
            self.__ky,
            self.__kz,
            self.__k2,
            self.__k4,
            self.__kx_mat,
            self.__ky_mat,
            self.__kz_mat,
        ) = prepare_fft(self.Nx, self.Ny, self.Nz, self.dx, self.dy, self.dz)

    def make_save_file(self) -> None:
        """
        Creates the directories needed to save the results of the simulation.

        Returns:
            None
        """
        msave.create_directory(self.save_path)
        msave.create_directory(f"{self.save_path}/{self.dirname}/res")

    def save_instance(self) -> None:
        """
        Saves the current state of the instance to a pickle file.

        Returns:
            None
        """
        res_dict = {}
        for key in self.__dict__:
            typename = type(self.__dict__[key]).__name__
            if typename != "function":
                res_dict[key] = self.__dict__[key]

        with open(f"{self.save_path}/{self.dirname}/instance.pickle", "wb") as file:
            pickle.dump(res_dict, file)

    def load_instance(self, full_dir_path: Union[str, None] = None) -> None:
        """
        Loads a previously saved instance from a pickle file.

        Args:
            full_dir_path (str, optional): The full directory path from which to load the instance. If None, the current directory is used.

        Returns:
            None
        """
        if full_dir_path is None:
            with open(f"{self.save_path}/{self.dirname}/instance.pickle", "rb") as file:
                loaded_instance = pickle.load(file)
            return loaded_instance
        else:
            with open(f"{full_dir_path}/instance.pickle", "rb") as file:
                loaded_instance = pickle.load(file)
            return loaded_instance

    def prepare_result_variables(self) -> None:
        """
        Prepares the variables used to store the results of the simulation.

        Returns:
            None
        """
        self.energy_g = cp.zeros(self.nstep) + cp.nan
        self.energy_el = cp.zeros(self.nstep) + cp.nan

        # derivatives of elastic energy
        self.delsdc = cp.zeros((self.Nx, self.Ny, self.Nz))
        # derivatives of free energy
        self.dfdcon = cp.zeros((self.Nx, self.Ny, self.Nz))
        # free energy
        self.g = cp.zeros((self.Nx, self.Ny, self.Nz))
        # elastic stress
        self.s = cp.zeros((self.Nx, self.Ny, self.Nz, 6))
        # elastic strain
        self.el = cp.zeros((self.Nx, self.Ny, self.Nz, 6))
        self.conk = cp.zeros((self.Nx, self.Ny, self.Nz, 6), dtype=cp.complex128)
        self.dgdck = cp.zeros((self.Nx, self.Ny, self.Nz, 6), dtype=cp.complex128)

        # set initial compositional noise
        self.con = add_initial_noise(self.Nx, self.Ny, self.Nz, self.c0, self.noise)

        cp.random.seed(123)

    def is_included_target_file_in_directory(
        self, directory: str, target_name: str
    ) -> bool:
        """
        Checks if a specific target file is present in the given directory.

        Args:
            directory (str): The directory to search in.
            target_name (str): The name of the target file to search for.

        Returns:
            bool: True if the target file is found, False otherwise.
        """
        for root, _, files in os.walk(directory):
            for file_name in files:
                if target_name == file_name:
                    return True
        return False

    def calculate_green_tensor(self) -> None:
        """
        Calculates the Green's tensor for the elasticity calculations, or loads a previously saved tensor if available.

        Returns:
            None
        """
        # the calculation of green tensor costs very high.
        # so save the green tensor result and
        # use previous one if once it is calculated.
        print(f"parameters: Nx{self.Nx}, T{int(self.T)}K, P{int(self.P)}Pa")
        msave.create_directory("resources")
        filename = f"tmatx_3d_feldspar_{int(self.Nx)}_{int(self.T)}K_{self.P}Pa.npy"

        if self.is_included_target_file_in_directory("resources", filename):
            print("using previous tmatx")
            self.tmatx = cp.asarray(np.load(f"resources/{filename}"))
        else:
            print("calculating tmatx")
            tmatx, omeg11 = green_tensor(
                cp.asnumpy(self.__kx),
                cp.asnumpy(self.__ky),
                cp.asnumpy(self.__kz),
                cp.asnumpy(self.__cp),
                cp.asnumpy(self.__cm),
            )
            np.save(f"resources/{filename}", tmatx)
            self.tmatx = cp.asarray(tmatx)

    def calculate_phase_filed(
        self, method: Literal["linear", "nonlinear"] = "linear"
    ) -> None:
        """
        Performs the phase field calculation using the specified method.

        Args:
            method (str, optional): The method for phase field calculation, either "linear" or "nonlinear". Default is "linear".

        Returns:
            None
        """
        roop_range = range(self.roop_start_from, self.nstep + 1)
        for istep in roop_range:
            if istep % 3 == 0:
                dtime = self.dtime * self.accel
            else:
                dtime = self.dtime
            if self.noise_per_step != 0:
                self.con = self.con = self.con + add_initial_noise(
                    self.Nx, self.Ny, self.Nz, 0, self.noise_per_step
                )

            self.iter = istep
            # print(istep)
            # Calculate derivatives of free energy and elastic energy
            self.delsdc, self.s, self.el = solve_elasticity(
                self.tmatx, self.__cm, self.__cp, self.ea, self.ei0, self.con, self.c0
            )

            # Assuming you have the get_free_energy and solve_elasticity_v2 functions
            self.dfdcon, self.g = get_free_energy(self.con, self.__wab, self.__wor)

            self.energy_g[istep - 1] = cp.sum(self.g)

            # バルク規格化
            self.con = (
                self.c0 - np.sum(self.con) / (self.Nx * self.Ny * self.Nz) + self.con
            )
            self.con = self.con * (1 + np.sum(self.con[self.con < 0]) / self.c0)
            self.con = self.con * (1 + np.sum(self.con[self.con > 1]) / self.c0)
            self.con[self.con < 0] = 0.00001
            self.con[self.con > 1] = 0.99999

            self.conk = cp.fft.fftn(self.con)
            self.dgdck = cp.fft.fftn(self.dfdcon + self.delsdc)
            # self.delsdck = cp.fft.fftn(self.delsdc)

            if method == "nonlinear":
                # diffusion term
                self.d = cp.abs(self.interdiffusion_fn(self.con))
                self.dk = cp.fft.fftn(self.d)
                term1 = (
                    cp.fft.fftn(
                        cp.fft.ifftn(1.0j * self.__kx_mat * self.dk)
                        * cp.fft.ifftn(1.0j * self.__kx_mat * self.dgdck)
                    )
                    + cp.fft.fftn(
                        cp.fft.ifftn(1.0j * self.__ky_mat * self.dk)
                        * cp.fft.ifftn(1.0j * self.__ky_mat * self.dgdck)
                    )
                    + cp.fft.fftn(
                        cp.fft.ifftn(1.0j * self.__kz_mat * self.dk)
                        * cp.fft.ifftn(1.0j * self.__kz_mat * self.dgdck)
                    )
                )

                self.gk = (
                    -self.__k2 * self.dgdck - self.__k4 * self.grad_coef * self.conk
                )

                term2 = cp.fft.fftn(cp.fft.ifftn(self.gk) * cp.fft.ifftn(self.dk))

                self.conk = self.conk + dtime * (term1 + term2)
                self.con = cp.real(cp.fft.ifftn(self.conk))
            else:
                # Time integration
                numer = dtime * self.mobility * self.__k2 * (self.dgdck)
                denom = (
                    1.0
                    + dtime * self.coefA * self.mobility * self.grad_coef * self.__k4
                )

                self.conk = (self.conk - numer) / denom
                self.con = cp.real(cp.fft.ifftn(self.conk))

            if self.nprint is not None:
                if (istep % self.nprint == 0) or (istep == 1):
                    # con_disp = np.flipud(cp.asnumpy(self.con.transpose()))
                    con_disp = self.con
                    # plt.imshow(con_disp)
                    # y
                    # ↑
                    # |
                    # + --→ x [100]
                    mplt.display_3d_matrix(cp.asnumpy(con_disp))
                    plt.show()

            if (istep % self.nsave == 0) or (istep == 1):
                np.save(
                    f"{self.save_path}/{self.dirname}/res/con_{istep}.npy",
                    cp.asnumpy(self.con),
                )
                np.save(
                    f"{self.save_path}/{self.dirname}/res/el_{istep}.npy",
                    cp.asnumpy(self.el),
                )
                np.save(
                    f"{self.save_path}/{self.dirname}/res/s_{istep}.npy",
                    cp.asnumpy(self.s),
                )


if __name__ == "__main__":
    from binary_coherent_3d import CoherentBinary3D, DataSet

    dataset = DataSet()

    feldspar = CoherentBinary3D(
        "result",
        stiffness=dataset.stiffness1,
        chemical_strain=dataset.ei0,
        method="linear",
    )

    feldspar.T = 500 + 273
    feldspar.c0 = 0.35
    feldspar.dtime = 0.1
    feldspar.nprint = 30
    feldspar.exec()

# %%
