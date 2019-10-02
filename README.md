## Package fpcross

> Solution of the multidimensional Fokker-Planck equation by fast and accurate tensor based methods with cross approximation in the tensor-train (TT) format.

#### Requirements

1. python 3.5;

1. standard python packages like numpy, scipy, matplotlib, etc. (all of them are included in [anaconda distribution](https://www.anaconda.com/download/));

1. python package [ttpy](https://github.com/oseledets/ttpy);

1. browser based gui for python [jupyter lab](https://github.com/jupyterlab/jupyterlab) (jupyter lab notebook).

#### Installation

1. install python 3.5 and standard python packages listed above. The best way is to use [anaconda distribution](https://www.anaconda.com/download/);

1. install ttpy python package according to instructions from [github repo](https://github.com/oseledets/ttpy);

1. download this repo and run `python setup.py install` from the root folder of the project.

  > To uninstall this package run `pip uninstall fpcross`.

1. Open jupyter lab files from the `examples` folder in the browser to run the code in the interactive mode.

#### Tests

Run `python ./tests/test.py` from the root folder of the project to perform all tests (`-v` option is available).

> TODO! Add more tests

#### Examples

All examples are performed as interactive browser-based jupyter lab notebooks (the corresponding package is included in anaconda distribution).

To work with example run in terminal `jupyter lab` (the corresponding page will be opened in the default web-browser), find in the directory tree the "fpcross/examples" folder, open the corresponding jupyter lab notebook and run all the cells one by one.

See `./examples/intertrain.ipynb` for interpolation submodule details, `./examples/main.ipynb` for example of solver usage and another jupyter lab notebooks in `./examples/` folder for examples of various special use cases.

#### Authors

- Andrei Chertkov (`andrei.chertkov@skolkovotech.ru`)

- Ivan Oseledets (`i.oseledets@skoltech.ru`)

#### Related publications

> TODO! Add publications
