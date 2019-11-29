import os
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
desc = 'Solution of the multidimensional Fokker-Planck equation by fast and accurate tensor based methods with cross approximation in the tensor-train (TT) format.'
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    desc_long = f.read()

setup(
    name = 'fpcross',
    version = '0.1',
    description=desc,
    long_description=desc_long,
    long_description_content_type='text/markdown',
    author='Andrei Chertkov',
    author_email='andrei.chertkov@skolkovotech.ru',
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
    keywords='Fokker-Planck equation stochastic differential equation probability density function low-rank representation tensor train format TT-decomposition cross approximation Chebyshev polynomial multivariate Ornstein-–Uhlenbeck process',
    packages=find_packages('fpcross', './fpcross/'),
    python_requires='>=3.7',
    install_requires=['ttpy'],
    project_urls={
        'Source': 'https://github.com/AndreiChertkov/fpcross',
    },
)

# See great example https://github.com/pypa/sampleproject
