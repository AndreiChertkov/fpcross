import os
import re
from setuptools import setup


def find_packages(package, basepath):
    packages = [package]
    for name in os.listdir(basepath):
        path = os.path.join(basepath, name)
        if not os.path.isdir(path):
            continue
        packages.extend(find_packages('%s.%s'%(package, name), path))
    return packages


here = os.path.abspath(os.path.dirname(__file__))
desc = 'Solver in the low-rank tensor-train format with cross approximation approach for solution of the multidimensional Fokker-Planck equation'
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    desc_long = f.read()


with open(os.path.join(here, 'fpcross/__init__.py'), encoding='utf-8') as f:
    text = f.read()
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
    version = version.group(1)


setup_args = dict(
    name='fpcross',
    version=version,
    description=desc,
    long_description=desc_long,
    long_description_content_type='text/markdown',
    author='Andrei Chertkov',
    author_email='andre.chertkov@gmail.com',
    url='https://github.com/AndreiChertkov/fpcross',
    classifiers=[
        'Development Status :: 3 - Alpha', # 4 - Beta, 5 - Production/Stable
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Framework :: Jupyter',
    ],
    keywords='Fokker-Planck equation low-rank representation tensor train format TT-decomposition cross approximation probability density estimation',
    packages=find_packages('fpcross', './fpcross/'),
    python_requires='>=3.7',
    project_urls={
        'Source': 'https://github.com/AndreiChertkov/fpcross',
    },
    license='MIT',
    license_files = ('LICENSE.txt',),
)


if __name__ == '__main__':
    setup(
        **setup_args,
        install_requires=[
            'teneva==0.11.6',
            'matplotlib==3.5.3',
            'numba==0.56.0',
            'numpy==1.22.0',
            'scipy==1.9.1',
            'tqdm==4.64.0',
            'llvmlite==0.39.0',
            'six==1.16.0',
        ],
        include_package_data=True)
