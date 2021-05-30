package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zungl2 generates an m-by-n complex matrix Q with orthonormal rows,
// which is defined as the first m rows of a product of k elementary
// reflectors of order n
//
//       Q  =  H(k)**H . . . H(2)**H H(1)**H
//
// as returned by ZGELQF.
func Zungl2(m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
	var one, zero complex128
	var i, j, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*m) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGL2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		return
	}

	if (*k) < (*m) {
		//        Initialise rows k+1:m to rows of the unit matrix
		for j = 1; j <= (*n); j++ {
			for l = (*k) + 1; l <= (*m); l++ {
				a.Set(l-1, j-1, zero)
			}
			if j > (*k) && j <= (*m) {
				a.Set(j-1, j-1, one)
			}
		}
	}

	for i = (*k); i >= 1; i-- {
		//        Apply H(i)**H to A(i:m,i:n) from the right
		if i < (*n) {
			Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
			if i < (*m) {
				a.Set(i-1, i-1, one)
				Zlarf('R', toPtr((*m)-i), toPtr((*n)-i+1), a.CVector(i-1, i-1), lda, toPtrc128(tau.GetConj(i-1)), a.Off(i+1-1, i-1), lda, work)
			}
			goblas.Zscal(toPtr((*n)-i), toPtrc128(-tau.Get(i-1)), a.CVector(i-1, i+1-1), lda)
			Zlacgv(toPtr((*n)-i), a.CVector(i-1, i+1-1), lda)
		}
		a.Set(i-1, i-1, one-tau.GetConj(i-1))

		//        Set A(i,1:i-1) to zero
		for l = 1; l <= i-1; l++ {
			a.Set(i-1, l-1, zero)
		}
	}
}
