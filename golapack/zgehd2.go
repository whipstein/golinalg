package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgehd2 reduces a complex general matrix A to upper Hessenberg form H
// by a unitary similarity transformation:  Q**H * A * Q = H .
func Zgehd2(n, ilo, ihi int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
	var alpha, one complex128
	var i int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=%v, n=%v", ilo, n)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=%v, ihi=%v, n=%v", ilo, ihi, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgehd2", err)
		return
	}

	for i = ilo; i <= ihi-1; i++ {
		//        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
		alpha = a.Get(i, i-1)
		alpha, *tau.GetPtr(i - 1) = Zlarfg(ihi-i, alpha, a.Off(min(i+2, n)-1, i-1).CVector(), 1)
		a.Set(i, i-1, one)

		//        Apply H(i) to A(1:ihi,i+1:ihi) from the right
		Zlarf(Right, ihi, ihi-i, a.Off(i, i-1).CVector(), 1, tau.Get(i-1), a.Off(0, i), work)

		//        Apply H(i)**H to A(i+1:ihi,i+1:n) from the left
		Zlarf(Left, ihi-i, n-i, a.Off(i, i-1).CVector(), 1, tau.GetConj(i-1), a.Off(i, i), work)

		a.Set(i, i-1, alpha)
	}

	return
}
