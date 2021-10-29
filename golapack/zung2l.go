package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zung2l generates an m by n complex matrix Q with orthonormal columns,
// which is defined as the last n columns of a product of k elementary
// reflectors of order m
//
//       Q  =  H(k) . . . H(2) H(1)
//
// as returned by ZGEQLF.
func Zung2l(m, n, k int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
	var one, zero complex128
	var i, ii, j, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

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
		gltest.Xerbla2("Zung2l", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	//     Initialise columns 1:n-k to columns of the unit matrix
	for j = 1; j <= n-k; j++ {
		for l = 1; l <= m; l++ {
			a.Set(l-1, j-1, zero)
		}
		a.Set(m-n+j-1, j-1, one)
	}

	for i = 1; i <= k; i++ {
		ii = n - k + i

		//        Apply H(i) to A(1:m-k+i,1:n-k+i) from the left
		a.Set(m-n+ii-1, ii-1, one)
		Zlarf(Left, m-n+ii, ii-1, a.CVector(0, ii-1, 1), tau.Get(i-1), a, work)
		goblas.Zscal(m-n+ii-1, -tau.Get(i-1), a.CVector(0, ii-1, 1))
		a.Set(m-n+ii-1, ii-1, one-tau.Get(i-1))

		//        Set A(m-k+i+1:m,n-k+i) to zero
		for l = m - n + ii + 1; l <= m; l++ {
			a.Set(l-1, ii-1, zero)
		}
	}

	return
}
