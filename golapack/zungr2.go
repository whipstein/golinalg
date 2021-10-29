package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungr2 generates an m by n complex matrix Q with orthonormal rows,
// which is defined as the last m rows of a product of k elementary
// reflectors of order n
//
//       Q  =  H(1)**H H(2)**H . . . H(k)**H
//
// as returned by ZGERQF.
func Zungr2(m, n, k int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
	var one, zero complex128
	var i, ii, j, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < m {
		err = fmt.Errorf("n < m: m=%v, n=%v", m, n)
	} else if k < 0 || k > m {
		err = fmt.Errorf("k < 0 || k > m: m=%v, k=%v", m, k)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zungr2", err)
		return
	}

	//     Quick return if possible
	if m <= 0 {
		return
	}

	if k < m {
		//        Initialise rows 1:m-k to rows of the unit matrix
		for j = 1; j <= n; j++ {
			for l = 1; l <= m-k; l++ {
				a.Set(l-1, j-1, zero)
			}
			if j > n-m && j <= n-k {
				a.Set(m-n+j-1, j-1, one)
			}
		}
	}

	for i = 1; i <= k; i++ {
		ii = m - k + i

		//        Apply H(i)**H to A(1:m-k+i,1:n-k+i) from the right
		Zlacgv(n-m+ii-1, a.CVector(ii-1, 0))
		a.Set(ii-1, n-m+ii-1, one)
		Zlarf(Right, ii-1, n-m+ii, a.CVector(ii-1, 0), tau.GetConj(i-1), a, work)
		goblas.Zscal(n-m+ii-1, -tau.Get(i-1), a.CVector(ii-1, 0))
		Zlacgv(n-m+ii-1, a.CVector(ii-1, 0))
		a.Set(ii-1, n-m+ii-1, one-tau.GetConj(i-1))

		//        Set A(m-k+i,n-k+i+1:n) to zero
		for l = n - m + ii + 1; l <= n; l++ {
			a.Set(ii-1, l-1, zero)
		}
	}

	return
}
