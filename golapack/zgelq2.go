package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelq2 computes an LQ factorization of a complex m-by-n matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a n-by-n orthogonal matrix;
//    L is an lower-triangular m-by-m matrix;
//    0 is a m-by-(n-m) zero matrix, if m < n.
func Zgelq2(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
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
		gltest.Xerbla([]byte("ZGELQ2"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i,i+1:n)
		Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
		alpha = a.Get(i-1, i-1)
		Zlarfg(toPtr((*n)-i+1), &alpha, a.CVector(i-1, minint(i+1, *n)-1), lda, tau.GetPtr(i-1))
		if i < (*m) {
			//           Apply H(i) to A(i+1:m,i:n) from the right
			a.Set(i-1, i-1, one)
			Zlarf('R', toPtr((*m)-i), toPtr((*n)-i+1), a.CVector(i-1, i-1), lda, tau.GetPtr(i-1), a.Off(i+1-1, i-1), lda, work)
		}
		a.Set(i-1, i-1, alpha)
		Zlacgv(toPtr((*n)-i+1), a.CVector(i-1, i-1), lda)
	}
}
