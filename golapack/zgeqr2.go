package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqr2 computes a QR factorization of a complex m-by-n matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a m-by-m orthogonal matrix;
//    R is an upper-triangular n-by-n matrix;
//    0 is a (m-n)-by-n zero matrix, if m > n.
func Zgeqr2(m, n int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
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
		gltest.Xerbla2("Zgeqr2", err)
		return
	}

	k = min(m, n)

	for i = 1; i <= k; i++ {
		//        Generate elementary reflector H(i) to annihilate A(i+1:m,i)
		*a.GetPtr(i-1, i-1), *tau.GetPtr(i - 1) = Zlarfg(m-i+1, a.Get(i-1, i-1), a.Off(min(i+1, m)-1, i-1).CVector(), 1)
		if i < n {
			//           Apply H(i)**H to A(i:m,i+1:n) from the left
			alpha = a.Get(i-1, i-1)
			a.Set(i-1, i-1, one)
			Zlarf(Left, m-i+1, n-i, a.Off(i-1, i-1).CVector(), 1, tau.GetConj(i-1), a.Off(i-1, i), work)
			a.Set(i-1, i-1, alpha)
		}
	}

	return
}
