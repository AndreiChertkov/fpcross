import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    '''
    Class for representation and construction of the
    uniform or Chebyshev multidimensional grid.

    Basic usage:
    1 Initialize class instance with grid parameters.
    2 Call "comp" for computation in given indices (or the full grid).
    3 Call "info" for demonstration of grid parameters.
    4 Call "copy" to obtain new instance with the same parameters.

    Advanced usage:
    - Call "rand" to obtain random points inside the grid limits.
    - Call "plot" for plot of some or all grid points (or random points).
    - Call "is_square" to check if grid is square.

    PROPS:

    d - dimension of the grid
    type: int, >= 1

    n - total number of points for each dimension
    type: ndarray [dimensions] of int, >= 2

    l - min-max values of variable for each dimension
    type: ndarray [dimensions, 2] of float, [:, 0] < [:, 1]

    h - grid steps (assuming uniformity) for each dimension
    type: ndarray [dimensions] of float, > 0

    kind - kind of the grid.
    type: str
    enum:
        - 'u' - uniform
        - 'c' - chebyshev

    n0 - average number of grid points
    type: int, >= 2

    l0 - average grid limits (min and max)
    type: ndarray [2] of float

    h0 - average grid step (assuming uniformity)
    type: float, > 0
    '''

    def __init__(self, d=None, n=2, l=[-1., 1.], kind='c'):
        '''
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
        * Note that [:, 0] (or [0]) are min and [:, 1] (or [1]) are max
        * values for each dimension. If it is 1D list or array (type1 or type2),
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
            n = np.array(n, dtype='int')
        if not isinstance(n, np.ndarray) or len(n.shape) != 1:
            raise ValueError('Invalid type for number of points (n).')
        if n.shape[0] != d:
            raise IndexError('Invalid shape for n parameter.')
        for i in range(d):
            if n[i] < 2:
                raise ValueError('Ivalid number of points (should be >= 2).')

        if isinstance(l, list):
            l = np.array(l)
        if not isinstance(l,np.ndarray) or len(l.shape) < 1 or len(l.shape) > 2:
            raise ValueError('Invalid type for limits (l).')
        if len(l.shape) == 1:
            l = np.repeat(l.reshape(1, -1), d, axis=0)
        if l.shape[0] != d or l.shape[1] != 2:
            raise IndexError('Invalid shape for l parameter.')
        for i in range(d):
            if l[i, 0] >= l[i, 1]:
                raise ValueError('Ivalid limits (min should be less of max).')

        if kind != 'u' and kind != 'c':
            raise ValueError('Invalid grid kind.')

        self.d = d
        self.n = n
        self.l = l
        self.h = (self.l[:, 1] - self.l[:, 0]) / (self.n - 1)
        self.kind = kind

        self.n0 = int(np.mean(self.n, axis=0))
        self.l0 = np.mean(self.l, axis=0)
        self.h0 = float(np.mean(self.h, axis=0))

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
        according to the grid limits l.

        INPUT:

        I - indices of grid points
        type1: None
        type2: int
        type3: list [number of points] of int
        type4: ndarray [number of points] of int
        type5: list [dimensions] of int
        type6: ndarray [dimensions] of int
        type7: list [dimensions, number of points] of int
        type8: ndarray [dimensions, number of points] of int
        * If is None (type1), then the full grid will be constructed.
        * Type2 is available only for the case of only one point in 1D.
        * Type3 and type4 are available only for 1D case.
        * Type5 and type6 - for only one point in the multidimensional case.

        is_ind - flag:
            True  - indices of points will be returned
            False - spatial grid points will be returned
        type: bool

        OUTPUT:

        I - (if is_ind == True) prepared indices of grid points
        type: ndarray [dimensions, number of points] of int

        X - (if is_ind == False) calculated grid points
        type: ndarray [dimensions, number of points] of float
        '''

        if I is None:
            I = [np.arange(self.n[d]).reshape(1, -1) for d in range(self.d)]
            I = np.meshgrid(*I, indexing='ij')
            I = np.array(I).reshape((self.d, -1), order='F')

        if isinstance(I, (int, float)):
            I = [int(I)]
        if isinstance(I, list):
            I = np.array(I, dtype='int')
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

        n_ = np.repeat(self.n.reshape((-1, 1)), I.shape[1], axis=1)
        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), I.shape[1], axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), I.shape[1], axis=1)

        if self.kind == 'u':
            t = I * 1. / (n_ - 1)
            X = t * (l2 - l1) + l1
        if self.kind == 'c':
            t = np.cos(np.pi * I / (n_ - 1))
            X = t * (l2 - l1) / 2. + (l2 + l1) / 2.

        return X

    def find(self, x):
        '''
        Find the nearest flatten grid index for the given spatial point.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        OUTPUT:

        i - flatten index of the grid point
        type: int

        TODO! Add support for calculation without explicit spatial grid.
        '''

        if isinstance(x, (int, float)):
            x = [float(x)]
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            raise ValueError('Invalid type of the spatial point.')
        if len(x.shape) != 1 or x.shape[0] != self.d:
            raise ValueError('Invalid shape of the spatial point.')

        l1 = self.l[:, 0]
        l2 = self.l[:, 1]
        n = self.n

        if self.kind == 'u':
            i = (x - l1) * (n - 1) / (l2 - l1)
        if self.kind == 'c':
            t = (2. * x - l2 - l1) / (l2 - l1)
            t[t > +1.] = +1.
            t[t < -1.] = -1.
            i = np.arccos(t) * (n - 1) / np.pi

        i = np.rint(i).astype(int)
        i[i <= 0] = 0.
        i[i >= n] = n[i >= n] - 1.

        # i = np.linalg.norm(self.X_hst - x, axis=0).argmin()
        return self.indf(i)

    def indm(self, i_f):
        '''
        Construct multi index from the given flatten index
        of the grid point.

        INPUT:

        i_f - flatten index of the grid point.
        type:  int, >= 0, < prod(n)

        OUTPUT:

        i_m - multi index of the grid point
        type:  ndarray [dimensions] of int, >= 0, < n
        '''

        i_m = np.unravel_index(i_f, self.n, order='F')
        return i_m

    def indf(self, i_m):
        '''
        Construct flatten index from the given multi index
        of the grid point.

        INPUT:

        i_m - multi index of the grid point
        type1:  list [dimensions] of int, >= 0, < n
        type2:  ndarray [dimensions] of int, >= 0, < n

        OUTPUT:

        i_f - flatten index of the grid point.
        type:  int, >= 0, < prod(n)
        '''

        i_f = np.ravel_multi_index(i_m, self.n, order='F')
        return i_f

    def rand(self, n):
        '''
        Generate random points inside the grid limits.
        * Uniform distribution is used.

        INPUT:

        n - total number of points
        type: int, > 0

        TODO: Maybe change function name (it sounds like random grid points).
        '''

        n = int(n)
        if n <= 0: raise ValueError('Invalid number of points.')

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
        '''

        s = '------------------ Grid\n'

        k = '???'
        if self.kind == 'u': k = 'Uniform'
        if self.kind == 'c': k = 'Chebyshev'
        s+= 'Kind             : %s\n'%k

        s+= 'Dimensions       : %-2d\n'%self.d

        s+= '%s             : '%('Mean' if not self.is_square() else '    ')
        s+= 'Poi %-3d | '%self.n0
        s+= 'Min %-6.3f | '%self.l0[0]
        s+= 'Max %-6.3f |\n'%self.l0[1]

        if not self.is_square():
            for i, [n, l] in enumerate(zip(self.n, self.l)):
                if i >= 5 and i < self.d - 5:
                    if i == 5:
                        s+= ' ...             : ...\n'
                    continue
                s+= 'Dim. # %-2d        : '%(i+1)
                s+= 'Poi %-3d | '%n
                s+= 'Min %-6.3f | '%l[0]
                s+= 'Max %-6.3f |\n'%l[1]

        if not s.endswith('\n'):
            s+= '\n'
        if is_print:
            print(s[:-1])
        else:
            return s

    def plot(self, I=None, n=None, x0=None):
        '''
        Plot the full grid or some grid points or some random points.
        * Only 2-dimensional case is supported.

        I - indices of grid points for plot
        * See description in Grid.comp function.

        n - total number of random points
        * See description in Grid.rand function.
        * If is set, then random points are used.


        x0 - special spatial point for present on the plot
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        '''

        if I is not None and n is not None:
            raise ValueError('Both I and n are set.')

        if isinstance(x0, (int, float)):
            x0 = [float(x0)]
        if isinstance(x0, list):
            x0 = np.array(x0)
        if x0 is not None and not isinstance(x0, np.ndarray):
            raise ValueError('Invalid type of the special spatial point.')
        if x0 is not None and (len(x0.shape) != 1 or x0.shape[0] != self.d):
            raise ValueError('Invalid shape of the special spatial point.')


        if self.d == 2:
            X = self.rand(n) if n is not None else self.comp(I)
            for k in range(X.shape[1]):
                x = X[:, k]
                plt.scatter(x[0], x[1])
                plt.text(x[0]+0.1, x[1]-0.1, '%d'%k)
            if x0 is not None:
                plt.scatter(
                    x0[0], x0[1], s=80., c='#8b1d1d', marker='*', alpha=0.9
                )
            plt.show()
        else:
            raise NotImplementedError('Invalid dimension for plot.')

    def is_out(self, x):
        '''
        Check if given point is out of the grid.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        OUTPUT:

        res - True if point is out of the grid and False otherwise
        type: bool
        '''

        if isinstance(x, (int, float)):
            x = [float(x)]
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            raise ValueError('Invalid type of the spatial point.')
        if len(x.shape) != 1 or x.shape[0] != self.d:
            raise ValueError('Invalid shape of the spatial point.')

        for i in range(self.d):
            if x[i] < self.l[i, 0]: return True
            if x[i] > self.l[i, 1]: return True
        return False

    def is_square(self, eps=1.E-20):
        '''
        Check if grid is square (all dimensions are equal in terms of
        numbers of grid points and limits).

        INPUT:

        eps - accuracy of check
        type: float, > 0

        OUTPUT:

        res - True if grid is square and False otherwise
        type: bool

        TODO: Maybe replace this function by the corresponding variable
              (in this case we need to be sure that the parameters n and l
              do not change externally).
        '''

        n0 = self.n0
        if np.max(np.abs(self.n - n0)) > eps: return False

        l0 = np.repeat(self.l0.reshape((1, -1)), self.d, axis=0)
        if np.max(np.abs(self.l - l0)) > eps: return False

        return True
