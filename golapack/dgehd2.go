package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgehd2 reduces a real general matrix A to upper Hessenberg form H by
// an orthogonal similarity transformation:  Q**T * A * Q = H .
func Dgehd2(n, ilo, ihi int, a *mat.Matrix, tau, work *mat.Vector) (err error) {
	var aii, one float64
	var i int

	one = 1.0

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
		gltest.Xerbla2("Dgehd2", err)
		return
	}

	for i = ilo; i <= ihi-1; i++ {
		//        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
		*a.GetPtr(i, i-1), *tau.GetPtr(i - 1) = Dlarfg(ihi-i, a.Get(i, i-1), a.Off(min(i+2, n)-1, i-1).Vector(), 1)
		aii = a.Get(i, i-1)
		a.Set(i, i-1, one)

		//        Apply H(i) to A(1:ihi,i+1:ihi) from the right
		Dlarf(Right, ihi, ihi-i, a.Off(i, i-1).Vector(), 1, tau.Get(i-1), a.Off(0, i), work)

		//        Apply H(i) to A(i+1:ihi,i+1:n) from the left
		Dlarf(Left, ihi-i, n-i, a.Off(i, i-1).Vector(), 1, tau.Get(i-1), a.Off(i, i), work)

		a.Set(i, i-1, aii)
	}

	return
}
