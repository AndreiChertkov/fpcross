import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    '''
    Class for representation and construction of the
    uniform or Chebyshev multidimensional grid.
    '''

    def __init__(self, d=None, n=2, l=[-1., 1.], kind='c'):
        '''
        INPUT:

        d - number of dimensions
        type: None or int, >= 1
        * If is None, then it will be recovered from n or l shape.

        n - total number of points for each dimension
        type: int or ndarray (or list) [dimensions] of int, >= 2
        * If has type int, then the same value will be used for each dimension.

        l - min-max values of variable for each dimension
        type: ndarray (or list) [dimensions, 2] of float, [:, 0] < [:, 1]
              or ndarray (or list) [2] of float, [0] < [1]
        * Note that [:, 0] (or [0]) are min and [:, 1] (or [1]) are max
        * values for each dimension.
        * If it is 1D array or list, then it will be used for each dimension.

        kind - kind of the grid.
        type: str, 'u' (uniform) or 'c' ('chebyshev')
        '''

        if d is not None:
            if not isinstance(n, (int, float)):
                raise ValueError('Invalid number of dimensions (d).')
            d = int(d)
        else:
            if isinstance(n, (list, np.ndarray)):
                d = len(n)
            elif isinstance(l, (list, np.ndarray)):
                d = len(l)
            else:
                raise ValueError('Dimension (d) is not set and can`t be calculated from n and l.')

        if isinstance(n, (int, float)):
            n = [int(n)] * (d or 1)
        if isinstance(n, list):
            n = np.array(n)
        if not isinstance(n, np.ndarray):
            raise IndexError('Invalid type for number of points (n).')

        if isinstance(l, list):
            l = np.array(l)
        if not isinstance(l, np.ndarray):
            raise IndexError('Invalid type for limits (l).')
        if len(l.shape) == 1:
            l = np.repeat(l.reshape(1, -1), d, axis=1)

        self.n = n
        self.l = l
        self.d = d or self.n.shape[0]

        if self.n.shape[0] != self.d or self.l.shape[0] != self.d:
            raise IndexError('Invalid shape for grid parameters.')

        self.kind = kind

    def copy(self):
        '''
        Create a copy of the class instance.

        OUTPUT:

        GR - new class instance
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
        type2: ndarray [dimensions, number of points] of int
        type3: list [dimensions, number of points] of int
        type4: ndarray [number of points] of int
        type5: list [number of points] of int
        type6: int
        * If is None (type1), then the full grid will be constructed.
        * Type4 and type5 are available only for 1D case.
        * Type6 is available only for 1D case and for only one point.

        OUTPUT:

        X - grid points
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
        if len(I.shape) == 1:
            I = I.reshape(1, -1)
        if I.shape[0] != self.d:
            raise ValueError('Invalid dimension of grid points.')
        if is_ind:
            return I

        n = np.repeat(self.n.reshape((-1, 1)), I.shape[1], axis=1)
        t = np.cos(np.pi * I / (n - 1))

        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), I.shape[1], axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), I.shape[1], axis=1)
        X = t * (l2 - l1) / 2. + (l2 + l1) / 2.

        return X

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
        s+= 'Kind             : %s\n'%('Chebyshev')
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

    def plot(self, I=None):
        '''
        Plot the full grid or some grid points.

        I - indices of points for plot
        type: ndarray [dimensions, number_of_ponts] of int
        * If it is not set, then the full grid will be used.
        '''

        if self.d == 2:
            X = self.comp(I)
            for k in range(X.shape[1]):
                x = X[:, k]
                plt.scatter(x[0], x[1])
                plt.text(x[0]+0.1, x[1]-0.1, '%d'%k)
            plt.show()
        else:
            raise NotImplementedError('Invalid dimension for plot.')
