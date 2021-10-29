package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgerq2 computes an RQ factorization of a real m by n matrix A:
// A = R * Q.
func Dgerq2(m, n int, a *mat.Matrix, tau, work *mat.Vector) (err error) {
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
		gltest.Xerbla2("Dgerq2", err)
		return
	}

	k = min(m, n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(m-k+i,1:n-k+i-1)
		*a.GetPtr(m-k+i-1, n-k+i-1), *tau.GetPtr(i - 1) = Dlarfg(n-k+i, a.Get(m-k+i-1, n-k+i-1), a.Vector(m-k+i-1, 0))

		//        Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right
		aii = a.Get(m-k+i-1, n-k+i-1)
		a.Set(m-k+i-1, n-k+i-1, one)
		Dlarf(Right, m-k+i-1, n-k+i, a.Vector(m-k+i-1, 0), tau.Get(i-1), a, work)
		a.Set(m-k+i-1, n-k+i-1, aii)
	}

	return
}
