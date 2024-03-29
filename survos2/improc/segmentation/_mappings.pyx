#cython: language_level=3
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False


import numpy as np
cimport numpy as np


def _rlabels(unsigned char[::1] y, unsigned int[::1] R, int ny, int nr, double min_ratio):
    cdef Py_ssize_t N = R.shape[0]
    cdef Py_ssize_t i, s, cmax, curr
    cdef unsigned char j, l
    cdef double smin

    cdef unsigned int[::1] sizes = np.zeros(nr, dtype=np.uint32)
    cdef unsigned int[:, ::1] counts = np.zeros((nr, ny), dtype=np.uint32)
    cdef unsigned char[::1] out = np.zeros(nr, dtype=np.uint8)

    for i in range(N):
        l = y[i]
        s = R[i]
        sizes[s] += 1
        if l > 0:
            counts[s, l] += 1

    for i in range(nr):
        cmax = 0
        smin = sizes[i] * min_ratio
        for j in range(1, ny):
            curr = counts[i, j]
            if curr > cmax and curr >= smin:
                cmax = curr
                out[i] = j

    return np.asarray(out)


def _rmeans(float[:, ::1] X, unsigned int[::1] R, int nr):
    cdef int n
    cdef int N = X.shape[0]  # length of the flattened feature array
    cdef int K = X.shape[1] # size of the feature stack (number of feature layers)
    cdef float[:, ::1] F = np.zeros((nr, K), np.float32)
    cdef int[::1] sizes = np.zeros(nr, np.int32)
    cdef unsigned int l

    for n in range(N):  # for each element in a single feature array
        l = R[n]        # get the superregion label for that location
        sizes[l] += 1   # increment the counter for that superregion label index 
        for z in range(K):  # for each layer in the feature stack
            F[l, z] += X[n, z]  # add the feature value at the location along the feature array and at a specific feature stack layer

    for n in range(nr):  # for each of the superpixel regions
        if sizes[n] > 0:  # if the superpixel has had any feature values added to it
            for z in range(K):  # for each individual layer
                F[n, z] /= sizes[n]  # get the mean of the features for that superpixel, keeping the feature layers separate

    return np.asarray(F)


def _rstats(float[:, ::1] X, unsigned int[::1] R, int nr):
    cdef int i, j, k
    cdef unsigned int r
    cdef int N = X.shape[0], K = X.shape[1]
    cdef int[::1] sizes = np.zeros(nr, dtype=np.int32)
    cdef float diff
    cdef float[:, ::1] means = np.zeros((nr, K), np.float32)
    cdef float[:, :, ::1] covars = np.zeros((nr, K, K), np.float32)

    for n in range(N):
        r = R[n]
        sizes[r] += 1
        for z in range(K):
            means[r, z] += X[n, z]

    for n in range(nr):
        for z in range(K):
            means[n, z] /= sizes[n]

    for i in range(N):
        r = R[i]
        for j in range(K):
            covars[r, j, j] += (X[i, j] - means[r, j]) * (X[i, j] - means[r, j]) / sizes[r]
            for k in range(j+1, K):
                diff = (X[i, j] - means[r, j]) * (X[i, k] - means[r, k]) / sizes[r]
                covars[r, j, k] += diff
                covars[r, k, j] += diff

    return np.asarray(means), np.asarray(covars)    