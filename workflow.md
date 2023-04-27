# workflow

> Workflow instructions for `fpcross` developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name fpcross python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate fpcross
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab
    ```

5. Install fpcross:
    ```bash
    python setup.py install
    ```

6. Reinstall fpcross (after updates of the code):
    ```bash
    clear && pip uninstall fpcross -y && python setup.py install
    ```

7. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name fpcross --all -y
    ```


## How to update the package version

1. Run the simple demo script:
    ```bash
    clear && python demo/demo.py
    ```

2. Run the complex demo script:
    ```bash
    clear && python demo/check.py
    ```

3. Update version (like `0.5.X`) in the file `fpcross/__init__.py`

4. Do commit `Update version (0.5.X)` and push

5. Upload new version to `pypi` (login: AndreiChertkov)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

6. Reinstall and check that installed version is new
    ```bash
    pip install --no-cache-dir --upgrade fpcross
    ```
