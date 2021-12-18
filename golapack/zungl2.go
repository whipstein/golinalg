package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungl2 generates an m-by-n complex matrix Q with orthonormal rows,
// which is defined as the first m rows of a product of k elementary
// reflectors of order n
//
//       Q  =  H(k)**H . . . H(2)**H H(1)**H
//
// as returned by ZGELQF.
func Zungl2(m, n, k int, a *mat.CMatrix, tau, work *mat.CVector) (err error) {
	var one, zero complex128
	var i, j, l int

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
		gltest.Xerbla2("Zungl2", err)
		return
	}

	//     Quick return if possible
	if m <= 0 {
		return
	}

	if k < m {
		//        Initialise rows k+1:m to rows of the unit matrix
		for j = 1; j <= n; j++ {
			for l = k + 1; l <= m; l++ {
				a.Set(l-1, j-1, zero)
			}
			if j > k && j <= m {
				a.Set(j-1, j-1, one)
			}
		}
	}

	for i = k; i >= 1; i-- {
		//        Apply H(i)**H to A(i:m,i:n) from the right
		if i < n {
			Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
			if i < m {
				a.Set(i-1, i-1, one)
				Zlarf(Right, m-i, n-i+1, a.Off(i-1, i-1).CVector(), a.Rows, tau.GetConj(i-1), a.Off(i, i-1), work)
			}
			a.Off(i-1, i).CVector().Scal(n-i, -tau.Get(i-1), a.Rows)
			Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
		}
		a.Set(i-1, i-1, one-tau.GetConj(i-1))

		//        Set A(i,1:i-1) to zero
		for l = 1; l <= i-1; l++ {
			a.Set(i-1, l-1, zero)
		}
	}

	return
}
