# fpcross


## Description

This python package, named **fpcross** (**F**okker **P**lanck **cross**-approximation), provides a solver in the low-rank tensor train format with cross approximation approach for solution of the multidimensional Fokker-Planck equation (FPE) of the form

```
d r(x, t) / d t = D delta( r(x, t) ) - div( f(x, t) r(x, t) ),
where r(x, 0) = r0(x).
```

The function f(x, t), its diagonal partial derivatives d f_i (x, t) / d x_i, initial condition r0(x) and scalar diffusion coefficient D should be known. The equation is solved from the initial moment (t = 0) to the user-specified moment (t), while the solutions obtained at each time step can be used if necessary. The resulting solution r(x, t) represents both the TT-tensor on the multidimensional Chebyshev grid and the Chebyshev interpolation coefficients in the TT-format, and therefore it can be quickly calculated at an arbitrary spatial point.


## Installation

The package can be installed via pip: `pip install fpcross` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [fpcross](https://github.com/AndreiChertkov/fpcross) and installed by `python setup.py install` command from the root folder of the project.

> It is highly recommended to create a virtual environment before installing (`conda create --name fpcross python=3.8` and then `conda activate fpcross`). The setup file contains the versions of the packages that were used when testing the software product, but it may work correct on newer versions.


## Usage

A compact example of using the solver for a user-defined FPE is provided in the script `demo/demo.py` (run it as `python demo/demo.py` from the root of the project).

The software product also implements classes for the model FPEs:
1. multidimensional simple diffusion problem (see `fpcross/equation_demo/equation_dif.py`);
2. multidimensional Ornstein-Uhlenbeck process (see `fpcross/equation_demo/equation_oup.py`);
3. 3-dimensional dumbbell model (see `fpcross/equation_demo/equation_dum.py`).

A demonstration of their solution is given in the script `demo/check.py` (run it as `python demo/check.py` from the root of the project).


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)


## Citation

If you find this approach and/or code useful in your research, please consider citing:

```bibtex
@article{chertkov2021solution,
    author    = {Chertkov, Andrei and Oseledets, Ivan},
    year      = {2021},
    title     = {Solution of the Fokker--Planck equation by cross approximation method in the tensor train format},
    journal   = {Frontiers in Artificial Intelligence},
    volume    = {4},
    issn      = {2624-8212},
    doi       = {10.3389/frai.2021.668215},
    url       = {https://www.frontiersin.org/article/10.3389/frai.2021.668215}
}
```
