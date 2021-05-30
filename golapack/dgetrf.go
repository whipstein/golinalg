package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgetrf computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//    A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 3 BLAS version of the algorithm.
func Dgetrf(m *int, n *int, a *mat.Matrix, lda *int, ipiv *[]int, info *int) {
	var one float64
	var i, iinfo, j, jb, nb int

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGETRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGETRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= minint(*m, *n) {
		//        Use unblocked code.
		Dgetrf2(m, n, a, lda, ipiv, info)
	} else {
		//        Use blocked code.
		for j = 1; j <= minint(*m, *n); j += nb {
			jb = minint(minint(*m, *n)-j+1, nb)

			//           Factor diagonal and subdiagonal blocks and test for exact
			//           singularity.
			_ipiv := (*ipiv)[j-1:]
			Dgetrf2(toPtr((*m)-j+1), &jb, a.Off(j-1, j-1), lda, &_ipiv, &iinfo)

			//           Adjust INFO and the pivot indices.
			if (*info) == 0 && iinfo > 0 {
				(*info) = iinfo + j - 1
			}
			for i = j; i <= minint(*m, j+jb-1); i++ {
				(*ipiv)[i-1] = j - 1 + (*ipiv)[i-1]
			}

			//           Apply interchanges to columns 1:J-1.
			Dlaswp(toPtr(j-1), a, lda, &j, toPtr(j+jb-1), ipiv, func() *int { y := 1; return &y }())

			if j+jb <= (*n) {
				//              Apply interchanges to columns J+JB:N.
				Dlaswp(toPtr((*n)-j-jb+1), a.Off(0, j+jb-1), lda, &j, toPtr(j+jb-1), ipiv, func() *int { y := 1; return &y }())

				//              Compute block row of U.
				goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, &jb, toPtr((*n)-j-jb+1), &one, a.Off(j-1, j-1), lda, a.Off(j-1, j+jb-1), lda)
				if j+jb <= (*m) {
					//                 Update trailing submatrix.
					goblas.Dgemm(mat.NoTrans, mat.NoTrans, toPtr((*m)-j-jb+1), toPtr((*n)-j-jb+1), &jb, toPtrf64(-one), a.Off(j+jb-1, j-1), lda, a.Off(j-1, j+jb-1), lda, &one, a.Off(j+jb-1, j+jb-1), lda)
				}
			}
		}
	}
}
