# Bootstrap in Quantum Mechanical system

## About the project
This repository follows the procedure from [David Berenstein et al. - Bootstrapping Simple QM Systems](https://arxiv.org/abs/2108.08757). The detail mathematical derivation can be found in ***bootstrap_derivation.pdf***, and the result will be discussed in ***bootstrap_result.pdf***.

---
## Installation

Since most of the code is written in `Sympy`, it is highly recommended to run with `Jupyter Notebook` or `Jupyter Lab`. The package we used is in ***requirements.txt***.

```bash
python -m pip install -r requirements.txt
```

---
## Algorithm

1. Construct a `class` with specific potential, also contains functions to compute **recursion relation** and the **determinant of sub-matrices**.
2. Find the interval for energy eigenvalues such that the **Hankel Matrix** is **positive-semi definite**. (Mainly processed by ***bootstrap_sympy.py***)
3. Plot the solved energy eigenvalues interval with different size of matrices.

---
## Get started

To get a feeling how we do the **Bootstrap Method**, you can run ***bootstrap_numpy.ipynb***, however the performance is restricted to due to the decimal precision.

Tha full main codes are written with ipynb, and run with `sympy`, select a specific potential and open the correspoding `ipynb` file.

- ***harmonic_sympy.ipynb*** : bootstrapping with harmonic potential $V(x)=kx^2$

- ***hydrogen_sympy.ipynb*** : bootstrapping with Coulomb potential $V(r)=-\frac{k}{r}$

- ***harmonic_sympy.ipynb*** : bootstrapping with Yukawa potential $V(r)=-\frac{k}{r}e^{-ar}$, but approximated to first order

The computed result can be load from the correspoding directories (***harmonic***, ***hydrogen***, ***yukawa_order1***)