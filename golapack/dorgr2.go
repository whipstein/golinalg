package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorgr2 generates an m by n real matrix Q with orthonormal rows,
// which is defined as the last m rows of a product of k elementary
// reflectors of order n
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by DGERQF.
func Dorgr2(m, n, k *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var one, zero float64
	var i, ii, j, l int

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla([]byte("DORGR2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		return
	}

	if (*k) < (*m) {
		//        Initialise rows 1:m-k to rows of the unit matrix
		for j = 1; j <= (*n); j++ {
			for l = 1; l <= (*m)-(*k); l++ {
				a.Set(l-1, j-1, zero)
			}
			if j > (*n)-(*m) && j <= (*n)-(*k) {
				a.Set((*m)-(*n)+j-1, j-1, one)
			}
		}
	}

	for i = 1; i <= (*k); i++ {
		ii = (*m) - (*k) + i

		//        Apply H(i) to A(1:m-k+i,1:n-k+i) from the right
		a.Set(ii-1, (*n)-(*m)+ii-1, one)
		Dlarf('R', toPtr(ii-1), toPtr((*n)-(*m)+ii), a.Vector(ii-1, 0), lda, tau.GetPtr(i-1), a, lda, work)
		goblas.Dscal((*n)-(*m)+ii-1, -tau.Get(i-1), a.Vector(ii-1, 0), *lda)
		a.Set(ii-1, (*n)-(*m)+ii-1, one-tau.Get(i-1))

		//        Set A(m-k+i,n-k+i+1:n) to zero
		for l = (*n) - (*m) + ii + 1; l <= (*n); l++ {
			a.Set(ii-1, l-1, zero)
		}
	}
}
