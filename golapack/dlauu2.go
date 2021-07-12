package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlauu2 computes the product U * U**T or L**T * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the unblocked form of the algorithm, calling Level 2 BLAS.
func Dlauu2(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	var upper bool
	var aii, one float64
	var i int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAUU2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the product U * U**T.
		for i = 1; i <= (*n); i++ {
			aii = a.Get(i-1, i-1)
			if i < (*n) {
				a.Set(i-1, i-1, goblas.Ddot((*n)-i+1, a.Vector(i-1, i-1), a.Vector(i-1, i-1)))
				err = goblas.Dgemv(NoTrans, i-1, (*n)-i, one, a.Off(0, i), a.Vector(i-1, i), aii, a.Vector(0, i-1, 1))
			} else {
				goblas.Dscal(i, aii, a.Vector(0, i-1, 1))
			}
		}

	} else {
		//        Compute the product L**T * L.
		for i = 1; i <= (*n); i++ {
			aii = a.Get(i-1, i-1)
			if i < (*n) {
				a.Set(i-1, i-1, goblas.Ddot((*n)-i+1, a.Vector(i-1, i-1, 1), a.Vector(i-1, i-1, 1)))
				err = goblas.Dgemv(Trans, (*n)-i, i-1, one, a.Off(i, 0), a.Vector(i, i-1, 1), aii, a.Vector(i-1, 0))
			} else {
				goblas.Dscal(i, aii, a.Vector(i-1, 0))
			}
		}
	}
}
