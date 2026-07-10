"""
Data Science Task # 08 - NumPy Fundamentals

Scenario: Clean and manipulate a dataset using NumPy, perform calculations,
and prepare it for further analysis.
"""

import numpy as np

# =============================================================================
# TASK 1: Creating NumPy Arrays
# =============================================================================
print("=" * 60)
print("TASK 1: Creating NumPy Arrays")
print("=" * 60)

# 1. Create a NumPy array from a Python list
ages = [23, 45, 34, 25, 42, 36, 29, 31]
ages_array = np.array(ages)
print("\n1. Ages array:", ages_array)

# 2. Create NumPy arrays of zeros and ones
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))
print("\n2a. Zeros array (3,4):\n", zeros_array)
print("\n2b. Ones array (2,5):\n", ones_array)

# 3. Create a NumPy array of random numbers between 0 and 1
random_array = np.random.rand(4, 3)
print("\n3. Random array (4,3) between 0-1:\n", random_array)


# =============================================================================
# TASK 2: Array Operations
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2: Array Operations")
print("=" * 60)

# 1. Basic arithmetic operations
arr1 = np.random.randint(1, 10, size=(2, 3))
arr2 = np.random.randint(1, 10, size=(2, 3))
print("\narr1:\n", arr1)
print("arr2:\n", arr2)

addition = arr1 + arr2
subtraction = arr1 - arr2
multiplication = arr1 * arr2
division = arr1 / arr2

print("\nAddition:\n", addition)
print("Subtraction:\n", subtraction)
print("Multiplication:\n", multiplication)
print("Division:\n", division)

# 2. Summary statistics of arr1
print("\n2. Summary statistics of arr1:")
print("Sum:", arr1.sum())
print("Mean:", arr1.mean())
print("Min:", arr1.min())
print("Max:", arr1.max())

# 3. Reshape arr1 into (3, 2)
reshaped_arr1 = arr1.reshape(3, 2)
print("\n3. Reshaped arr1 (3,2):\n", reshaped_arr1)


# =============================================================================
# TASK 3: Indexing and Slicing
# =============================================================================
print("\n" + "=" * 60)
print("TASK 3: Indexing and Slicing")
print("=" * 60)

# 1. Access and modify elements
print("\nOriginal arr1:\n", arr1)
print("Second element of first row:", arr1[0, 1])
arr1[0, 1] = 20
print("arr1 after modifying element to 20:\n", arr1)

# 2. Extract elements greater than 5
greater_than_5 = arr1[arr1 > 5]
print("\nElements in arr1 greater than 5:", greater_than_5)


# =============================================================================
# TASK 4: Broadcasting
# =============================================================================
print("\n" + "=" * 60)
print("TASK 4: Broadcasting")
print("=" * 60)

# 1. Broadcasting arithmetic
arr3 = np.array([[1], [2], [3]])          # shape (3,1)
arr4 = np.array([[4, 5, 6]])              # shape (1,3)
broadcast_sum = arr3 + arr4
print("\narr3 (3,1):\n", arr3)
print("arr4 (1,3):\n", arr4)
print("arr3 + arr4 (broadcasted, 3x3):\n", broadcast_sum)

# 2. Scalar operation
arr3_times_10 = arr3 * 10
print("\narr3 * 10:\n", arr3_times_10)


# =============================================================================
# TASK 5: Array Concatenation and Splitting
# =============================================================================
print("\n" + "=" * 60)
print("TASK 5: Array Concatenation and Splitting")
print("=" * 60)

# 1. Concatenate arr1 and arr2 along the first axis (axis=0)
concatenated = np.concatenate((arr1, arr2), axis=0)
print("\nConcatenated arr1 & arr2 (axis=0):\n", concatenated)

# 2. Split the concatenated array into two arrays along the second axis (axis=1)
# Note: concatenated has 3 columns, which doesn't divide evenly by 2,
# so array_split is used (allows unequal-sized splits).
split_arrays = np.array_split(concatenated, 2, axis=1)
print("\nSplit array part 1:\n", split_arrays[0])
print("Split array part 2:\n", split_arrays[1])


# =============================================================================
# TASK 6: Statistics and Linear Algebra
# =============================================================================
print("\n" + "=" * 60)
print("TASK 6: Statistics and Linear Algebra")
print("=" * 60)

# 1. Dot product of two 2D arrays
mat1 = np.random.randint(1, 10, size=(2, 3))
mat2 = np.random.randint(1, 10, size=(3, 2))
dot_product = np.dot(mat1, mat2)
print("\nmat1 (2,3):\n", mat1)
print("mat2 (3,2):\n", mat2)
print("Dot product (mat1 . mat2):\n", dot_product)

# 2. Eigenvalues and eigenvectors of a 2x2 matrix
square_matrix = np.array([[4, 2],
                           [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
print("\n2x2 matrix:\n", square_matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 3. Inverse of a 3x3 matrix
matrix_3x3 = np.array([[2, 1, 1],
                        [1, 3, 2],
                        [1, 0, 0]])
inverse_matrix = np.linalg.inv(matrix_3x3)
print("\n3x3 matrix:\n", matrix_3x3)
print("Inverse of matrix:\n", inverse_matrix)

# Verify: matrix * inverse should be (approximately) identity
print("\nVerification (matrix @ inverse ≈ identity):\n",
      np.round(matrix_3x3 @ inverse_matrix, decimals=10))
