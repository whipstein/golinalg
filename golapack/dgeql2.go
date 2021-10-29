package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeql2 computes a QL factorization of a real m by n matrix A:
// A = Q * L.
func Dgeql2(m, n int, a *mat.Matrix, tau, work *mat.Vector) (err error) {
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
		gltest.Xerbla2("Dgeql2", err)
		return
	}

	k = min(m, n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(1:m-k+i-1,n-k+i)
		*a.GetPtr(m-k+i-1, n-k+i-1), *tau.GetPtr(i - 1) = Dlarfg(m-k+i, a.Get(m-k+i-1, n-k+i-1), a.Vector(0, n-k+i-1, 1))

		//        Apply H(i) to A(1:m-k+i,1:n-k+i-1) from the left
		aii = a.Get(m-k+i-1, n-k+i-1)
		a.Set(m-k+i-1, n-k+i-1, one)
		Dlarf(Left, m-k+i, n-k+i-1, a.Vector(0, n-k+i-1, 1), tau.Get(i-1), a, work)
		a.Set(m-k+i-1, n-k+i-1, aii)
	}

	return
}
