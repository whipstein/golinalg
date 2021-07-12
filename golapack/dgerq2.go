package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgerq2 computes an RQ factorization of a real m by n matrix A:
// A = R * Q.
func Dgerq2(m, n *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var aii, one float64
	var i, k int

	one = 1.0

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGERQ2"), -(*info))
		return
	}

	k = min(*m, *n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(m-k+i,1:n-k+i-1)
		Dlarfg(toPtr((*n)-k+i), a.GetPtr((*m)-k+i-1, (*n)-k+i-1), a.Vector((*m)-k+i-1, 0), lda, tau.GetPtr(i-1))

		//        Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right
		aii = a.Get((*m)-k+i-1, (*n)-k+i-1)
		a.Set((*m)-k+i-1, (*n)-k+i-1, one)
		Dlarf('R', toPtr((*m)-k+i-1), toPtr((*n)-k+i), a.Vector((*m)-k+i-1, 0), lda, tau.GetPtr(i-1), a, lda, work)
		a.Set((*m)-k+i-1, (*n)-k+i-1, aii)
	}
}
