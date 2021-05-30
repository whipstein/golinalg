package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeqr2 computes a QR factorization of a real m-by-n matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a m-by-m orthogonal matrix;
//    R is an upper-triangular n-by-n matrix;
//    0 is a (m-n)-by-n zero matrix, if m > n.
func Dgeqr2(m, n *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
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
		gltest.Xerbla([]byte("DGEQR2"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i+1:m,i)
		Dlarfg(toPtr((*m)-i+1), a.GetPtr(i-1, i-1), a.Vector(minint(i+1, *m)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		if i < (*n) {
			//           Apply H(i) to A(i:m,i+1:n) from the left
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)
			Dlarf('L', toPtr((*m)-i+1), toPtr((*n)-i), a.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(i-1, i+1-1), lda, work)
			a.Set(i-1, i-1, aii)
		}
	}
}