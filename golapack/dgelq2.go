package golapack

import (
	"fmt"

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
func Dgelq2(m, n int, a *mat.Matrix, tau, work *mat.Vector) (err error) {
	var aii, one float64
	var i, k int

	one = 1.0

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgelq2", err)
		return
	}

	k = min(m, n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i,i+1:n)
		*a.GetPtr(i-1, i-1), *tau.GetPtr(i - 1) = Dlarfg(n-i+1, a.Get(i-1, i-1), a.Vector(i-1, min(i+1, n)-1))
		if i < m {
			//           Apply H(i) to A(i+1:m,i:n) from the right
			aii = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)
			Dlarf(Right, m-i, n-i+1, a.Vector(i-1, i-1), tau.Get(i-1), a.Off(i, i-1), work)
			a.Set(i-1, i-1, aii)
		}
	}

	return
}
