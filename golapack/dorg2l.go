package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorg2l generates an m by n real matrix Q with orthonormal columns,
// which is defined as the last n columns of a product of k elementary
// reflectors of order m
//
//       Q  =  H(k) . . . H(2) H(1)
//
// as returned by DGEQLF.
func Dorg2l(m, n, k *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var one, zero float64
	var i, ii, j, l int

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
		gltest.Xerbla([]byte("DORG2L"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	//     Initialise columns 1:n-k to columns of the unit matrix
	for j = 1; j <= (*n)-(*k); j++ {
		for l = 1; l <= (*m); l++ {
			a.Set(l-1, j-1, zero)
		}
		a.Set((*m)-(*n)+j-1, j-1, one)
	}

	for i = 1; i <= (*k); i++ {
		ii = (*n) - (*k) + i

		//        Apply H(i) to A(1:m-k+i,1:n-k+i) from the left
		a.Set((*m)-(*n)+ii-1, ii-1, one)
		Dlarf('L', toPtr((*m)-(*n)+ii), toPtr(ii-1), a.Vector(0, ii-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a, lda, work)
		goblas.Dscal(toPtr((*m)-(*n)+ii-1), toPtrf64(-tau.Get(i-1)), a.Vector(0, ii-1), toPtr(1))
		a.Set((*m)-(*n)+ii-1, ii-1, one-tau.Get(i-1))

		//        Set A(m-k+i+1:m,n-k+i) to zero
		for l = (*m) - (*n) + ii + 1; l <= (*m); l++ {
			a.Set(l-1, ii-1, zero)
		}
	}
}
