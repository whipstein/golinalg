package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlauu2 computes the product U * U**H or L**H * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the unblocked form of the algorithm, calling Level 2 BLAS.
func Zlauu2(uplo byte, n *int, a *mat.CMatrix, lda, info *int) {
	var upper bool
	var one complex128
	var aii float64
	var i int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAUU2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the product U * U**H.
		for i = 1; i <= (*n); i++ {
			aii = real(a.Get(i-1, i-1))
			if i < (*n) {
				a.SetRe(i-1, i-1, aii*aii+real(goblas.Zdotc((*n)-i, a.CVector(i-1, i+1-1), *lda, a.CVector(i-1, i+1-1), *lda)))
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
				err = goblas.Zgemv(NoTrans, i-1, (*n)-i, one, a.Off(0, i+1-1), *lda, a.CVector(i-1, i+1-1), *lda, complex(aii, 0), a.CVector(0, i-1), 1)
				Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
			} else {
				goblas.Zdscal(i, aii, a.CVector(0, i-1), 1)
			}
		}

	} else {
		//        Compute the product L**H * L.
		for i = 1; i <= (*n); i++ {
			aii = real(a.Get(i-1, i-1))
			if i < (*n) {
				a.SetRe(i-1, i-1, aii*aii+real(goblas.Zdotc((*n)-i, a.CVector(i+1-1, i-1), 1, a.CVector(i+1-1, i-1), 1)))
				Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
				err = goblas.Zgemv(ConjTrans, (*n)-i, i-1, one, a.Off(i+1-1, 0), *lda, a.CVector(i+1-1, i-1), 1, complex(aii, 0), a.CVector(i-1, 0), *lda)
				Zlacgv(toPtr(i-1), a.CVector(i-1, 0), lda)
			} else {
				goblas.Zdscal(i, aii, a.CVector(i-1, 0), *lda)
			}
		}
	}
}
