import os
from setuptools import setup

def find_packages(package, basepath):
    packages = [package]
    for name in os.listdir(basepath):
        path = os.path.join(basepath, name)
        if not os.path.isdir(path): continue
        packages.extend(find_packages('%s.%s'%(package, name), path))
    return packages

setup(
    name = 'fpcross',
    version = '0.1',
    packages = find_packages('fpcross', './fpcross/'),
    include_package_data = True,
    requires = ['python (>= 3.5)'],
    description  = 'Solution of the multidimensional Fokker-Planck equation by fast and accurate tensor based methods with cross approximation in the tensor-train (TT) format.',
    long_description =  open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    author = 'Andrei Chertkov and Ivan Oseledets',
    author_email = 'andrei.chertkov@skolkovotech.ru, i.oseledets@skoltech.ru',
    url = 'https://github.com/AndreiChertkov/fpcross',
    download_url = 'https://github.com/AndreiChertkov/fpcross',
    #license = 'BSD License',
    keywords = 'Fokker-Planck equation, stochastic differential equation, probability density function, low-rank representation, tensor train format, TT-decomposition, cross approximation, Chebyshev polynomial, multivariate Ornstein-–Uhlenbeck process',
    classifiers = [
    'Environment :: Web Environment',
    'Intended Audience :: Developers',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.5',
    'Topic :: Solvers',
    #'License :: OSI Approved :: BSD License',
    ],
)
