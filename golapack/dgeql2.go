package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeql2 computes a QL factorization of a real m by n matrix A:
// A = Q * L.
func Dgeql2(m, n *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
	var aii, one float64
	var i, k int

	one = 1.0

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQL2"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(1:m-k+i-1,n-k+i)
		Dlarfg(toPtr((*m)-k+i), a.GetPtr((*m)-k+i-1, (*n)-k+i-1), a.Vector(0, (*n)-k+i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))

		//        Apply H(i) to A(1:m-k+i,1:n-k+i-1) from the left
		aii = a.Get((*m)-k+i-1, (*n)-k+i-1)
		a.Set((*m)-k+i-1, (*n)-k+i-1, one)
		Dlarf('L', toPtr((*m)-k+i), toPtr((*n)-k+i-1), a.Vector(0, (*n)-k+i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a, lda, work)
		a.Set((*m)-k+i-1, (*n)-k+i-1, aii)
	}
}
