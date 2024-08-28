# %%
from coLamB import CoherentBinary3D, DataSet

dataset = DataSet()
# %%

# -----------------------------------
# for linear case
# -----------------------------------
feldspar = CoherentBinary3D(
    "result",
    stiffness=dataset.stiffness,
    chemical_strain=dataset.ei0,
    method="linear",
)

feldspar.T = 500 + 273
feldspar.c0 = 0.35
feldspar.dtime = 0.1
feldspar.nprint = 300
feldspar.exec()

# %%
# -----------------------------------
# for nonlinear case (extremely slow)
# -----------------------------------
feldspar = CoherentBinary3D(
    "result",
    stiffness=dataset.stiffness,
    chemical_strain=dataset.ei0,
    method="nonlinear",
)

feldspar.T = 500 + 273
feldspar.c0 = 0.35
feldspar.dtime = 0.001
feldspar.nprint = 300
feldspar.interdiffusion_fn = dataset.interdiffusion_fn
feldspar.exec()

# %%
