package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeql2 computes a QL factorization of a complex m by n matrix A:
// A = Q * L.
func Zgeql2(m, n int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
	var alpha, one complex128
	var i, k int

	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zgeql2", err)
		return
	}

	k = min(m, n)

	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(1:m-k+i-1,n-k+i)
		alpha = a.Get(m-k+i-1, n-k+i-1)
		alpha, *tau.GetPtr(i - 1) = Zlarfg(m-k+i, alpha, a.CVector(0, n-k+i-1, 1))

		//        Apply H(i)**H to A(1:m-k+i,1:n-k+i-1) from the left
		a.Set(m-k+i-1, n-k+i-1, one)
		Zlarf(Left, m-k+i, n-k+i-1, a.CVector(0, n-k+i-1, 1), tau.GetConj(i-1), a, work)
		a.Set(m-k+i-1, n-k+i-1, alpha)
	}

	return
}
