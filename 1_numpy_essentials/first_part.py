# %% Imports
import numpy as np

# %% create simple numpy array
array = [1,2,4]

numpy_array = np.array(array)
numpy_array

# %% create nested numpy array
nested_array = [[1,2],[4,5]]

numpy_nested_array = np.array(nested_array)
numpy_nested_array

# %% see array type
numpy_array.dtype, numpy_nested_array.dtype

# %% Other ways to create numpy arrays
zeros = np.zeros((2,2))
ones = np.ones((2,2))
random = np.empty((2,2))
arange = np.arange(10)
identity = np.eye(2,2)

zeros, ones, random, arange, identity

# %% Specify type
array = [1,2,4]
int_array = np.array(array, dtype=np.int32)
float_array = np.array(array, dtype=np.float32)

int_array.dtype, int_array, float_array.dtype, float_array

# %% Convert between type
array = [1,2,4]
int_array = np.array(array)
float_array = int_array.astype(np.float32)

int_array.dtype, int_array, float_array.dtype, float_array

# %% Scalar multiplication
array = np.array([1,2,4])

2* array, 0.4* array, array ** 3, array / 1.5

# %% Indexing and slicing
array = np.arange(10)
slice = array[5:8]
array[5:8] = -1
slice[0] = -4

slice, array

# %% Copy data
array = np.arange(10)

slice = array[5:8].copy()
array[5:8] = -1
slice[0] = -4

slice, array

# %% slicing 3d
array = np.random.rand(27).reshape(3,3,3)


array, array[0], array[:1], array[1:, 1], array[0, :1, 1:]

# %% Slicing array with another array (~ negates boolean expression)
str_array = np.array(['a', 'b', 'c', 'd', 'a'])
array = np.random.randn(5,5)

array[str_array != 'a', 2], array[~(str_array == 'a'), 2:]
# %% Using masks (We can use & or | )
mask = (str_array=='a') | (str_array == 'b')

array[mask]

# %% Other slicing options
array[array < 0] = 0
array[str_array == "c"] = 7

array

# %% Possible loops
zeros = np.zeros((6,6))

for i in range(6):
    zeros[i] = i

zeros

# %% Transposing
array = np.arange(15).reshape((3,5))
array.T, np.dot(array.T, array), array.T @ array

# %% Swapping axis
array = np.arange(15).reshape((3,5))
arrayT = array.swapaxes(0,1)
array, arrayT, array.T

# %% Specifying seed
np.random.seed(seed = 12345)

np.random.standard_normal((2,3))

# %% Creating a specific number generator
rng = np.random.default_rng(seed=12345)

rng.standard_normal((2,3))

# %% Unary Functions
array = np.arange(15)

np.sqrt(array), np.exp(array)

# %% Binary Functions
x, y = rng.standard_normal(4), rng.standard_normal(4)

x, y, np.maximum(x,y), np.minimum(x,y)

# %% uFunc (several outputs)
array = rng.standard_normal(4) * 5

remainder, whole_part = np.modf(array)

array, remainder, whole_part

# %% Selecting output
out = np.zeros_like(remainder)

np.add(whole_part, 1, out=out)

whole_part, out

# %% Conditional Logic
x, y = np.arange(5) * 2, np.arange(5) * 4 
condition = np.array([True, False, True, True, False])

x, y, np.where(condition, x, y)

# %% Mathematical functions
array = rng.standard_normal((5, 4))

array.mean(), array.std(), array.sum(), array.sum(axis=0), array.sum(axis=1), 

# %% Cumulative Sum
array = np.arange(1, 11).reshape((5,2))
array, array.cumsum(), array.cumprod(), array.cumsum(axis=0), array.cumsum(axis=1)

# %% Boolean operations
array = np.array([True, False, True])

array, array.any(), array.all()

# %% Sorting
array = rng.standard_normal(6)

array.sort()
array

# %% Unique
array = np.array([1,2,1,2,4])

np.unique(array)

# %% Belongs to set
array = np.array([1,2,4,5,6,1,2])

np.in1d(array, [1,4,5])

# %% AND, OR, In X but not in Y, XOR
x = np.array([1,2,4])
y = np.array([4,5,6])

np.intersect1d(x,y), np.union1d(x,y), np.setdiff1d(x,y), np.setxor1d(x,y)

# %% Saving arrays
x = np.array([1,2,4])
y = np.array([4,5,6])

np.save("x.npy", x)
x_1 = np.load("x.npy")

np.savez("x_y.npz", a=x, b=y)
z = (x_2, y_2) = np.load("x_y.npz")

np.savez_compressed("x_y_compressed.npz", a=x, b=y)
z_1 = (x_3, y_3) = np.load("x_y_compressed.npz")

x, y, x_1, z, x_2, y_2, z["a"], z["b"], x_3, y_3, z_1["a"], z_1["b"] 

# %% Linear Algebra
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])

x.dot(y), np.dot(x,y), x@y, x.T.dot(x)

# %% Matrix functions
from numpy import linalg as l

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7]])

x, 
np.diag(x),  # Diagonal  
np.trace(x), # Sum of Diagonal 
l.det(y), # Determinant 
l.eig(y), # Eigenvalue 
l.pinv(y), # Pseudo-Inverse 
l.qr(x), # QR decomposition
l.svd(x), # singular value decomposition
l.solve(y, [1, 4]) # solves Ax = b -> A = y and b = [1,4]
l.lstsq(x, [1, 4]) # Solves the least square solution to Ax = b

# %% Random walk
import matplotlib.pyplot as plt
nsteps = 1000
rng = np.random.default_rng(seed=12)
draws = rng.integers(0, 2, size=nsteps)
steps = np.where(draws == 0, 1, -1)

walk = steps.cumsum()

plt.plot(walk[:100])
plt.show()

walk.min(), walk.max(), (np.abs(walk) >= 10).argmax()

# %% several random walks
nsteps = 1000
nWalks = 5000

rng = np.random.default_rng(seed=12)
draws = rng.integers(0,2, size=(nWalks,nsteps))

steps = np.where(draws == 0, 1, -1)
walks = steps.cumsum(axis=1)

above30 = (np.abs(walks) > 30).any(axis=1)
crossIndex = (np.abs(walks[above30] >= 30)).argmax(axis=1)
walks.min(), walks.max(), above30.sum(), crossIndex.mean()