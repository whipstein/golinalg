package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dorg2r generates an m by n real matrix Q with orthonormal columns,
// which is defined as the first n columns of a product of k elementary
// reflectors of order m
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by DGEQRF.
func Dorg2r(m, n, k *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var one, zero float64
	var i, j, l int

	one = 1.0
	zero = 0.0

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) > (*m) {
		(*info) = -2
	} else if (*k) < 0 || (*k) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORG2R"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	//     Initialise columns k+1:n to columns of the unit matrix
	for j = (*k) + 1; j <= (*n); j++ {
		for l = 1; l <= (*m); l++ {
			a.Set(l-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}

	for i = (*k); i >= 1; i-- {
		//        Apply H(i) to A(i:m,i:n) from the left
		if i < (*n) {
			a.Set(i-1, i-1, one)
			Dlarf('L', toPtr((*m)-i+1), toPtr((*n)-i), a.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(i-1, i+1-1), lda, work)
		}
		if i < (*m) {
			goblas.Dscal(toPtr((*m)-i), toPtrf64(-tau.Get(i-1)), a.Vector(i+1-1, i-1), toPtr(1))
		}
		a.Set(i-1, i-1, one-tau.Get(i-1))

		//        Set A(1:i-1,i) to zero
		for l = 1; l <= i-1; l++ {
			a.Set(l-1, i-1, zero)
		}
	}
}
