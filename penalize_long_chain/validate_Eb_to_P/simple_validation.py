# This script validate the idea of mapping the bond breaking energy to the probability of bond breaking discussed in '/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/log/mapping_E_to_P.md'

import numpy as np
import matplotlib.pyplot as plt

A = 10.0
E_ref = 300.0
k = 1000.0

# A = 4.30724212053
# E_ref = 1625.23671637
# k = 698.365425223

BOND_ENERGIES = {
    "H-H": 436,
    "C-H": 413,
    "N-H": 391,
    "O-H": 463,
    "C-C": 346,
    "C-N": 305,
    "C-O": 358,
    # "N-N": 163,
    # "N-O": 201,
    "O-O": 146,
    "C=C": 602,
    "O=O": 498,
    "C=O": 732,
    # "N=O": 607,
    # "N=N": 418,
    # "C=N": 615,
    "C#C": 835,
    "C#O": 1072,
    "N#N": 945,
    "C#N": 887,
}

def calculate_p_break_from_energy(E_b, A=A, E_ref=E_ref, k=k):
    Arr = A * np.exp(-np.asarray(E_b) / E_ref)
    return 1 / (1 + k * np.exp(-Arr))

E_b = np.linspace(50, 1100, 1000)
P_break = calculate_p_break_from_energy(E_b)

plt.plot(E_b, P_break)

for i, (bond, energy) in enumerate(BOND_ENERGIES.items()):
    p_break = calculate_p_break_from_energy(energy)
    plt.scatter(energy, p_break, s=20)
    plt.annotate(
        bond,
        (energy, p_break),
        xytext=(0, 7 + 6 * (i % 3)),
        textcoords="offset points",
        ha="center",
        fontsize=7,
    )

plt.xlabel(r"$E_{B.D.E}$ (kJ/mol)")
plt.ylabel(r"$P_{\mathrm{break},b}$")
# plt.title(r"$P_{\mathrm{break},b}=\frac{1}{1+k e^{-Arr_b}},\quad Arr_b=Ae^{-E_b/E_{ref}}$")
plt.grid(True)
plt.tight_layout()
plt.show()
