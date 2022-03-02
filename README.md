# Package fpcross


## Description

Solution of the multidimensional Fokker-Planck equation (FPE) of the form

d r(x, t) / d t = D delta( r(x, t) ) - div( f(x, t) r(x, t) ),
r(x, 0) = r0(x),

with known f(x, t) and its partial derivatives, initial condition r0(x) and scalar diffusion coefficient D by fast and accurate tensor based methods with cross approximation in the tensor-train (TT) format.

The working version of the code with all numerical examples is in the following colab notebooks:
- [fpcross_np](https://colab.research.google.com/drive/1-1atifKoTE8nNSggsD42KFr28xk6MqIj?usp=sharing);
  > "Numpy" version of the code. This code is computationally inefficient (except the case of one-dimensional tasks) and is provided for illustration only.
- [fpcross_tt](https://colab.research.google.com/drive/19IfqOoexSr42xo_GCV3eZZpvTYg2YJhw?usp=sharing);
  > Basic version of the code (TT-format is used).
- See also folder `results_for_paper_fpcross` with scripts for generation plots for the paper.


## Authors

- Andrei Chertkov (a.chertkov@skoltech.ru);
- Ivan Oseledets (i.oseledets@skoltech.ru).


## Citation

If you find this code useful in your research, please consider citing:

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
