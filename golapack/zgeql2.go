package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeql2 computes a QL factorization of a complex m by n matrix A:
// A = Q * L.
func Zgeql2(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
	var alpha, one complex128
	var i, k int

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZGEQL2"), -(*info))
		return
	}

	k = min(*m, *n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(1:m-k+i-1,n-k+i)
		alpha = a.Get((*m)-k+i-1, (*n)-k+i-1)
		Zlarfg(toPtr((*m)-k+i), &alpha, a.CVector(0, (*n)-k+i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))

		//        Apply H(i)**H to A(1:m-k+i,1:n-k+i-1) from the left
		a.Set((*m)-k+i-1, (*n)-k+i-1, one)
		Zlarf('L', toPtr((*m)-k+i), toPtr((*n)-k+i-1), a.CVector(0, (*n)-k+i-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(i-1)), a, lda, work)
		a.Set((*m)-k+i-1, (*n)-k+i-1, alpha)
	}
}
