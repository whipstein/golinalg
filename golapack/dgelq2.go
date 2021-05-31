package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelq2 computes an LQ factorization of a real m-by-n matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a n-by-n orthogonal matrix;
//    L is an lower-triangular m-by-m matrix;
//    0 is a m-by-(n-m) zero matrix, if m < n.
func Dgelq2(m, n *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, info *int) {
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
		gltest.Xerbla([]byte("DGELQ2"), -(*info))
		return
	}

	k = minint(*m, *n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i,i+1:n)
		Dlarfg(toPtr((*n)-i+1), a.GetPtr(i-1, i-1), a.Vector(i-1, minint(i+1, *n)-1), lda, tau.GetPtr(i-1))
		if i < (*m) {
			//           Apply H(i) to A(i+1:m,i:n) from the right
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)
			Dlarf('R', toPtr((*m)-i), toPtr((*n)-i+1), a.Vector(i-1, i-1), lda, tau.GetPtr(i-1), a.Off(i+1-1, i-1), lda, work)
			a.Set(i-1, i-1, aii)
		}
	}
}
