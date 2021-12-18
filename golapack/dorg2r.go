package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorg2r generates an m by n real matrix Q with orthonormal columns,
// which is defined as the first n columns of a product of k elementary
// reflectors of order m
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by DGEQRF.
func Dorg2r(m, n, k int, a *mat.Matrix, tau, work *mat.Vector) (err error) {
	var one, zero float64
	var i, j, l int

	one = 1.0
	zero = 0.0

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: n=%v, m=%v", n, m)
	} else if k < 0 || k > n {
		err = fmt.Errorf("k < 0 || k > n: k=%v, n=%v", k, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dorg2r", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	//     Initialise columns k+1:n to columns of the unit matrix
	for j = k + 1; j <= n; j++ {
		for l = 1; l <= m; l++ {
			a.Set(l-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}

	for i = k; i >= 1; i-- {
		//        Apply H(i) to A(i:m,i:n) from the left
		if i < n {
			a.Set(i-1, i-1, one)
			Dlarf(Left, m-i+1, n-i, a.Off(i-1, i-1).Vector(), 1, tau.Get(i-1), a.Off(i-1, i), work)
		}
		if i < m {
			a.Off(i, i-1).Vector().Scal(m-i, -tau.Get(i-1), 1)
		}
		a.Set(i-1, i-1, one-tau.Get(i-1))

		//        Set A(1:i-1,i) to zero
		for l = 1; l <= i-1; l++ {
			a.Set(l-1, i-1, zero)
		}
	}

	return
}
