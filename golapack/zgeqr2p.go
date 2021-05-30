package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgeqr2p computes a QR factorization of a complex m-by-n matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a m-by-m orthogonal matrix;
//    R is an upper-triangular n-by-n matrix with nonnegative diagonal
//    entries;
//    0 is a (m-n)-by-n zero matrix, if m > n.
func Zgeqr2p(m, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, info *int) {
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
		gltest.Xerbla([]byte("ZGEQR2P"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i+1:m,i)
		Zlarfgp(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.CVector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		if i < (*n) {
			//           Apply H(i)**H to A(i:m,i+1:n) from the left
			alpha = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)
			Zlarf('L', toPtr((*m)-i+1), toPtr((*n)-i), a.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(i-1)), a.Off(i-1, i+1-1), lda, work)
			a.Set(i-1, i-1, alpha)
		}
	}
}