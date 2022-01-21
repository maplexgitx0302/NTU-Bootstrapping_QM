# Bootstrap in Quantum Mechanical system

## About the project
This repository follows the procedure from [David Berenstein et al. - Bootstrapping Simple QM Systems](https://arxiv.org/abs/2108.08757). The detail mathematical derivation can be found in <font color=#008000>**bootstrap_derivation.pdf**</font>, and the result will be discussed in <font color=#008000>**bootstrap_result.pdf**</font>.

---
## Installation

Since most of the code is written in `Sympy`, it is highly recommended to run with `Jupyter Notebook` or `Jupyter Lab`. The package we used is in <font color=#008000>**requirements.txt**</font>.

```bash
python -m pip install -r requirements.txt
```

---
## Algorithm

1. Construct a `class` with specific potential, also contains functions to compute **recursion relation** and the **determinant of sub-matrices**.
2. Find the interval for energy eigenvalues such that the matrix $M$ with element $M_{ij}=\langle X^{i+j}\rangle$ is **positive-semi definite**. (Mainly processed by <font color=#008000>**bootstrap_sympy.py**</font>)
3. Plot the solved energy eigenvalues interval with different size of matrices.

---
## Get started

To get a feeling how we do the **Bootstrap Method**, you can run <font color=#008000>**bootstrap_numpy.ipynb**</font>, however the performance is restricted to due to the decimal precision.

Tha full main codes are written with ipynb, and run with `sympy`, select a specific potential and open the correspoding `ipynb` file.

- <font color=#008000>**harmonic_sympy.ipynb**</font> : bootstrapping with harmonic potential $V(x)=kx^2$

- <font color=#008000>**hydrogen_sympy.ipynb**</font> : bootstrapping with Coulomb potential $V(r)=-\frac{k}{r}$

- <font color=#008000>**harmonic_sympy.ipynb**</font> : bootstrapping with Yukawa potential $V(r)=-\frac{k}{r}e^{-ar}$, but approximated to first order

The computed result can be load from the correspoding directories (<font color=#008000>**harmonic**</font>, <font color=#008000>**hydrogen**</font>, <font color=#008000>**yukawa_order1**</font>)