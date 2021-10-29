package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgerq2 computes an RQ factorization of a complex m by n matrix A:
// A = R * Q.
func Zgerq2(m, n int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
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
		gltest.Xerbla2("Zgerq2", err)
		return
	}

	k = min(m, n)
	for i = k; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        A(m-k+i,1:n-k+i-1)
		Zlacgv(n-k+i, a.CVector(m-k+i-1, 0))
		alpha = a.Get(m-k+i-1, n-k+i-1)
		alpha, *tau.GetPtr(i - 1) = Zlarfg(n-k+i, alpha, a.CVector(m-k+i-1, 0))

		//        Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right
		a.Set(m-k+i-1, n-k+i-1, one)
		Zlarf(Right, m-k+i-1, n-k+i, a.CVector(m-k+i-1, 0), tau.Get(i-1), a, work)
		a.Set(m-k+i-1, n-k+i-1, alpha)
		Zlacgv(n-k+i-1, a.CVector(m-k+i-1, 0))
	}

	return
}
