# Raman Mixture Analysis with NNLS - NonNegative Least Squares (No PCA)
# Purpose: Quantify 2-compound in MRT spectra (with 2 intermediates)
"""
Filename:exp-NNLS.py
Author: Lucille Kuster
Date created: 2026-02-03
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.interpolate import interp1d
import os


# ============================
# PATH + FILE NAMES
# ============================

path = "/Users/lucille/Desktop/test"

MIX_FILE   = "MRT-allcycles-LK018.txt"
PURE_FILES = [
    "Int-6-DFT-rR785nm.txt",
    "IntDimer-21-DFT-rR785nm.txt",
    "4,4-bipyr-ramanexp-lk014.txt",
    "b2O4H4-ramanexp-lk014.txt"
]

# Build full paths
mix_path = os.path.join(path, MIX_FILE)
pure_paths = [os.path.join(path, f) for f in PURE_FILES]

# ============================
# LOAD MIXTURE DATA
# ============================
mix_data = pd.read_csv(mix_path, sep=r"\s+", header=None)
w_mix = mix_data.iloc[:, 0].values
mix_spectra = mix_data.iloc[:, 1:].values

# ============================
# LOAD AND INTERPOLATE PURE SPECTRA
# ============================
pure_interp_list = []

for p in pure_paths:
    pure = pd.read_csv(p, sep=r"\s+", header=None)
    w = pure.iloc[:, 0].values
    y = pure.iloc[:, 1].values
    f = interp1d(w, y, kind="linear", fill_value="extrapolate")
    pure_interp = f(w_mix)
    pure_interp_list.append(pure_interp)

# ============================
# NORMALIZATION
# ============================
def normalize_area(y):
    area = np.trapezoid(y)
    if area == 0:
        return y
    return y / area

# Normalize pure spectra
pure_norm_list = [normalize_area(p) for p in pure_interp_list]

# Normalize mixture spectra
mix_n = np.array([
    normalize_area(mix_spectra[:, i])
    for i in range(mix_spectra.shape[1])
]).T

# ============================
# BUILD NNLS MATRIX
# ============================
A = np.column_stack(pure_norm_list)
n_pures = len(pure_norm_list)

# ============================
# NNLS FIT
# ============================
print("\n-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-")
print("|     NNLS MIXTURE COMPOSITION     |")
print("-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-‾-_-\n")

coeffs_list = []

for i in range(mix_n.shape[1]):
    y = mix_n[:, i]

    # NNLS
    coeffs, _ = nnls(A, y)

    # Scale fit
    fit = A @ coeffs
    scale = np.dot(y, fit) / np.dot(fit, fit)
    coeffs *= scale

    # Fractions
    total = np.sum(coeffs)
    if total > 0:
        frac = coeffs / total
    else:
        frac = np.zeros_like(coeffs)

    coeffs_list.append(frac)

    # Print results based on number of pure spectra
    output = f"Spectrum {i+1:02d} : "
    for j, fname in enumerate(PURE_FILES):
        output += f"{os.path.splitext(fname)[0]} = {frac[j]:.3f}   "
    print(output)

coeffs_array = np.array(coeffs_list)

# ============================
# PLOT FITS - (test - optional)
# ============================
# for i in range(mix_n.shape[1]):
#     y = mix_n[:, i]
#     coeffs = coeffs_array[i]

#     fit = np.sum([coeffs[j]*pure_norm_list[j] for j in range(n_pures)], axis=0)

#     plt.figure(figsize=(8,5))
#     plt.plot(w_mix, y, label="MRT", linewidth=2)
#     plt.plot(w_mix, fit, "--", label="NNLS Fit")

    # Plot each pure contribution
#     for j in range(n_pures):
#         plt.plot(w_mix, coeffs[j]*pure_norm_list[j], ":", label=f"{os.path.splitext(PURE_FILES[j])[0]} contrib")

#     plt.xlabel("Raman shift (cm$^{-1}$)")
#     plt.ylabel("Normalized intensity")
#     plt.title(f"Spectrum {i+1}")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# ============================
# BAR PLOT (COMPOSITION) - optional
# ============================

# Colors for each pure spectrum
# colors = ["#FBB4B4",  # pastel pink for Pure1
#           "#B4DFFB",  # pastel blue for Pure2
#           "#D9B4FB",  # pastel purple for Pure3
#           "#FF6961"]  # pastel red for Magic blue exp


# plt.figure(figsize=(8,5))
# x = np.arange(1, mix_n.shape[1]+1)
# width = 0.8 / n_pures  # adjust bar width dynamically

# for j in range(n_pures):
#     plt.bar(x - 0.4 + j*width + width/2,
#             coeffs_array[:, j],
#             width=width,
#             label=os.path.splitext(PURE_FILES[j])[0],
#             color=colors[j])  # <-- assign color here

# plt.xlabel("MRT cycle number")
# plt.ylabel("Fraction")
# plt.title("Mixture Composition (NNLS)")
# plt.legend()
# plt.tight_layout()
# plt.show()

