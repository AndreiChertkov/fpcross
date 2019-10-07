import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    '''
    Class for representation and construction of the
    uniform or Chebyshev multidimensional grid.
    '''

    def __init__(self, d=None, n=2, l=[-1., 1.], kind='c'):
        '''
        Init grid parameters.

        INPUT:

        d - number of dimensions
        type1: None
        type2: int, >= 1
        * If is None (type1), then it will be recovered from n or l shape.
        * If is set (type2), then n and l will be extended (if required)
        * according to the number of dimensions.

        n - total number of points for each dimension
        type1: int
        type2: list [dimensions] of int, >= 2
        type3: ndarray [dimensions] of int, >= 2
        * If is int (type1), then it will be used for each dimension.

        l - min-max values of variable for each dimension
        type1: list [2] of float, [0] < [1]
        type2: ndarray [2] of float, [0] < [1]
        type3: list [dimensions, 2] of float, [:, 0] < [:, 1]
        type4: ndarray [dimensions, 2] of float, [:, 0] < [:, 1]
        * Note that [:, 0] (or [0]) are min
        * and [:, 1] (or [1]) are max values for each dimension.
        * If it is 1D list or array (type1 or type2),
        * then the same values will be used for each dimension.

        kind - kind of the grid.
        type: str
        enum:
            - 'u' - uniform
            - 'c' - chebyshev
        '''

        if d is None:
            d = 1
            if isinstance(n, (list, np.ndarray)):
                if len(n) > 0:
                    d = len(n)
            elif isinstance(l, (list, np.ndarray)):
                if len(l) > 0 and isinstance(l[0], (list, np.ndarray)):
                    d = len(l)
        else:
            if not isinstance(d, (int, float)) or d < 1:
                raise ValueError('Invalid number of dimensions (d).')
            d = int(d)

        if isinstance(n, (int, float)):
            n = [int(n)] * d
        if isinstance(n, list):
            n = np.array(n)
        if not isinstance(n, np.ndarray):
            raise ValueError('Invalid type for number of points (n).')
        if n.shape[0] != d:
            raise IndexError('Invalid shape for n parameter.')

        if isinstance(l, list):
            l = np.array(l)
        if not isinstance(l, np.ndarray):
            raise ValueError('Invalid type for limits (l).')
        if len(l.shape) == 1:
            l = np.repeat(l.reshape(1, -1), d, axis=0)
        if l.shape[0] != d or l.shape[1] != 2:
            raise IndexError('Invalid shape for l parameter.')

        self.d = d
        self.n = n
        self.l = l
        self.kind = kind

    def copy(self):
        '''
        Create a copy of the grid.

        OUTPUT:

        GR - new class instance with the same parameters
        type: Grid
        '''

        return Grid(self.d, self.n.copy(), self.l.copy(), self.kind)

    def comp(self, I=None, is_ind=False):
        '''
        Compute grid points for given multi-indices.
        In case of the Chebyshev multidimensional grid points for every axis k
        are calculated as x = cos(i[k]*pi/(n[k]-1)), where n[k] is a total
        number of points for selected axis k, and then these points are scaled
        according to the interpolation limits l.

        INPUT:

        I - indices of grid points
        type1: None
        type2: int
        type3: list [number of points] of int
        type4: ndarray [number of points] of int
        type5: list [dimensions, number of points] of int
        type6: ndarray [dimensions, number of points] of int
        * If is None (type1), then the full grid will be constructed.
        * Type2 is available only for 1D case and for only one point.
        * Type3 and type4 are available only for 1D case of for only one point.

        is_ind - flag:
            True  - indices of points will be returned
            False - spatial grid points will be returned
        type: bool

        OUTPUT:

        I - (if is_ind == True) indices of grid points
        type: ndarray [dimensions, number of points] of int

        X - (if is_ind == False) grid points
        type: ndarray [dimensions, number of points] of float
        '''

        if I is None:
            I = [np.arange(self.n[d]).reshape(1, -1) for d in range(self.d)]
            I = np.meshgrid(*I, indexing='ij')
            I = np.array(I).reshape((self.d, -1), order='F')
        if isinstance(I, (int, float)):
            I = [I]
        if isinstance(I, list):
            I = np.array(I)
        if not isinstance(I, np.ndarray):
            raise ValueError('Invalid grid points.')
        if len(I.shape) == 1 and self.d >= 2:
            I = I.reshape(-1, 1)
        if len(I.shape) == 1 and self.d == 1:
            I = I.reshape(1, -1)
        if I.shape[0] != self.d:
            raise ValueError('Invalid dimension for grid points.')
        if is_ind:
            return I

        n0 = np.repeat(self.n.reshape((-1, 1)), I.shape[1], axis=1)
        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), I.shape[1], axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), I.shape[1], axis=1)

        if self.kind == 'u':
            t = I * 1. / (n0 - 1)
            X = t * (l2 - l1) + l1
        if self.kind == 'c':
            t = np.cos(np.pi * I / (n0 - 1))
            X = t * (l2 - l1) / 2. + (l2 + l1) / 2.

        return X

    def rand(self, n):
        '''
        Generate random points inside the grid.
        * Uniform distribution is used.

        INPUT:

        n - total number of points
        type: int, > 0
        '''

        n = int(n)
        if n < 0: raise ValueError('Invalid number of points (n).')

        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), n, axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), n, axis=1)
        return l1 + np.random.random((self.d, n)) * (l2 - l1)

    def info(self, is_print=True):
        '''
        Present info about the grid.

        INPUT:

        is_print - flag:
            True  - print string info
            False - return string info
        type: bool

        OUTPUT:

        s - (if is_print == False) string with info
        type: str

        TODO! Cut lim info if too many dimensions?
        '''

        s = '------------------ Grid\n'

        k = '???'
        if self.kind == 'u': k = 'Uniform'
        if self.kind == 'c': k = 'Chebyshev'
        s+= 'Kind             : %s\n'%k

        s+= 'Dimension        : %d\n'%self.d

        for i, [n, l] in enumerate(zip(self.n, self.l)):
            s+= 'D%-2d              : '%(i+1)
            s+= 'Poi %-3d | '%n
            s+= 'Min %-6.3f | '%l[0]
            s+= 'Max %-6.3f |\n'%l[1]

        s+='------------------\n'

        if is_print:
            print(s)
        else:
            return s

    def plot(self, I=None, n=None):
        '''
        Plot the full grid or some grid points or some random points.
        * Only 2-dimensional case is supported.

        I - indices of grid points for plot
        * See description in Grid.comp function.

        n - total number of points
        * See description in Grid.rand function.
        * If is set, then random points are used.

        '''

        if self.d == 2:
            X = self.rand(n) if n is not None else self.comp(I)
            for k in range(X.shape[1]):
                x = X[:, k]
                plt.scatter(x[0], x[1])
                plt.text(x[0]+0.1, x[1]-0.1, '%d'%k)
            plt.show()
        else:
            raise NotImplementedError('Invalid dimension for plot.')
