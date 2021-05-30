package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgerq2 computes an RQ factorization of a complex m by n matrix A:
// A = R * Q.
func Zgerq2(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
	var alpha, one complex128
	var i, k int

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZGERQ2"), -(*info))
		return
	}

	k = minint(*m, *n)
	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(m-k+i,1:n-k+i-1)
		Zlacgv(toPtr((*n)-k+i), a.CVector((*m)-k+i-1, 0), lda)
		alpha = a.Get((*m)-k+i-1, (*n)-k+i-1)
		Zlarfg(toPtr((*n)-k+i), &alpha, a.CVector((*m)-k+i-1, 0), lda, tau.GetPtr(i-1))

		//        Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right
		a.Set((*m)-k+i-1, (*n)-k+i-1, one)
		Zlarf('R', toPtr((*m)-k+i-1), toPtr((*n)-k+i), a.CVector((*m)-k+i-1, 0), lda, tau.GetPtr(i-1), a, lda, work)
		a.Set((*m)-k+i-1, (*n)-k+i-1, alpha)
		Zlacgv(toPtr((*n)-k+i-1), a.CVector((*m)-k+i-1, 0), lda)
	}
}
