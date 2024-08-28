# %%
# import numpy as np
import cupy as cp


class CDArray: ...


def get_free_energy(
    con: CDArray, w_ab: float, w_or: float
) -> tuple[[CDArray, CDArray]]:
    """
    Computes the free energy and its derivative with respect to concentration for a given material system.

    Args:
        con (CDArray): The concentration field of the precipitate phase.
        w_ab (float): The interaction parameter for component A and B.
        w_or (float): The interaction parameter for the original phase.

    Returns:
        tuple[CDArray, CDArray]: A tuple containing:
            - dfdcon: The derivative of the free energy with respect to the concentration field.
            - g: The free energy density field.
    """

    w = con * w_ab + (1 - con) * w_or
    ww = w_ab - w_or

    def get_dfdcon(con: CDArray) -> CDArray:
        dfdcon = (
            w * (1 - 2 * con) + con * (1 - con) * ww + (cp.log(con) - cp.log(1 - con))
        )
        return dfdcon

    min_c = 0.001
    max_c = 0.999

    dfdcon = cp.zeros(con.shape)
    dfdcon = get_dfdcon(con)
    dfdcon[con < min_c] = (get_dfdcon(min_c))[con < min_c]
    dfdcon[con > max_c] = (get_dfdcon(max_c))[con > max_c]

    g = w * con * (1 - con) + (con * cp.log(con) + (1 - con) * cp.log(1 - con))

    return dfdcon, g


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    cp.random.seed(seed=0)
    con = cp.random.rand(4, 4)
    R = 8.31
    P = 10**5
    T = 600 + 273
    w_ab = (22820 - 6.3 * T + 0.461 * P / 10**5) / (R * T)
    w_or = (19550 - 10.5 * T + 0.327 * P / 10**5) / (R * T)

    dg, g = get_free_energy(con, w_ab, w_or)
    print(dg)

    print("----------------------")

    con[0, 0] = -1.0
    con[0, 1] = -1.0
    con[1, 0] = 2.0
    con[1, 1] = 2.0
    dg, g = get_free_energy(con, w_ab, w_or)
    print(dg)

# %%
