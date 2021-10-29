package eig

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zunt03 compares two unitary matrices U and V to see if their
// corresponding rows or columns span the same spaces.  The rows are
// checked if RC = 'R', and the columns are checked if RC = 'C'.
//
// RESULT is the maximum of
//
//    | V*V' - I | / ( MV ulp ), if RC = 'R', or
//
//    | V'*V - I | / ( MV ulp ), if RC = 'C',
//
// and the maximum over rows (or columns) 1 to K of
//
//    | U(i) - S*V(i) |/ ( N ulp )
//
// where abs(S) = 1 (chosen to minimize the expression), U(i) is the
// i-th row (column) of U, and V(i) is the i-th row (column) of V.
func zunt03(rc byte, mu, mv, n, k int, u, v *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (result float64, err error) {
	var s, su, sv complex128
	var one, res1, res2, ulp, zero float64
	var i, irc, j, lmx int

	zero = 0.0
	one = 1.0

	//     Check inputs
	if rc == 'R' {
		irc = 0
	} else if rc == 'C' {
		irc = 1
	} else {
		irc = -1
	}
	if irc == -1 {
		err = fmt.Errorf("irc == -1: irc=%v", irc)
	} else if mu < 0 {
		err = fmt.Errorf("mu < 0: mu=%v", mu)
	} else if mv < 0 {
		err = fmt.Errorf("mv < 0: mv=%v", mv)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > max(mu, mv) {
		err = fmt.Errorf("k < 0 || k > max(mu, mv): k=%v, mu=%v, mv=%v", k, mu, mv)
	} else if (irc == 0 && u.Rows < max(1, mu)) || (irc == 1 && u.Rows < max(1, n)) {
		err = fmt.Errorf("(irc == 0 && u.Rows < max(1, mu)) || (irc == 1 && u.Rows < max(1, n)): irc=%v, u.Rows=%v, mu=%v, n=%v", irc, u.Rows, mu, n)
	} else if (irc == 0 && v.Rows < max(1, mv)) || (irc == 1 && v.Rows < max(1, n)) {
		err = fmt.Errorf("(irc == 0 && v.Rows < max(1, mv)) || (irc == 1 && v.Rows < max(1, n)): irc=%v, v.Rows=%v, mv=%v, n=%v", irc, v.Rows, mv, n)
	}
	if err != nil {
		gltest.Xerbla2("zunt03", err)
		return
	}

	//     Initialize result
	result = zero
	if mu == 0 || mv == 0 || n == 0 {
		return
	}

	//     Machine constants
	ulp = golapack.Dlamch(Precision)

	if irc == 0 {
		//        Compare rows
		res1 = zero
		for i = 1; i <= k; i++ {
			lmx = goblas.Izamax(n, u.CVector(i-1, 0))
			if v.Get(i-1, lmx-1) == complex(zero, 0) {
				sv = complex(one, 0)
			} else {
				sv = complex(v.GetMag(i-1, lmx-1), 0) / v.Get(i-1, lmx-1)
			}
			if u.Get(i-1, lmx-1) == complex(zero, 0) {
				su = complex(one, 0)
			} else {
				su = complex(u.GetMag(i-1, lmx-1), 0) / u.Get(i-1, lmx-1)
			}
			s = sv / su
			for j = 1; j <= n; j++ {
				res1 = math.Max(res1, cmplx.Abs(u.Get(i-1, j-1)-s*v.Get(i-1, j-1)))
			}
		}
		res1 = res1 / (float64(n) * ulp)

		//        Compute orthogonality of rows of V.
		res2 = zunt01('R', mv, n, v, work, lwork, rwork)

	} else {
		//        Compare columns
		res1 = zero
		for i = 1; i <= k; i++ {
			lmx = goblas.Izamax(n, u.CVector(0, i-1, 1))
			if v.Get(lmx-1, i-1) == complex(zero, 0) {
				sv = complex(one, 0)
			} else {
				sv = complex(v.GetMag(lmx-1, i-1), 0) / v.Get(lmx-1, i-1)
			}
			if u.Get(lmx-1, i-1) == complex(zero, 0) {
				su = complex(one, 0)
			} else {
				su = complex(u.GetMag(lmx-1, i-1), 0) / u.Get(lmx-1, i-1)
			}
			s = sv / su
			for j = 1; j <= n; j++ {
				res1 = math.Max(res1, cmplx.Abs(u.Get(j-1, i-1)-s*v.Get(j-1, i-1)))
			}
		}
		res1 = res1 / (float64(n) * ulp)

		//        Compute orthogonality of columns of V.
		res2 = zunt01('C', n, mv, v, work, lwork, rwork)
	}

	result = math.Min(math.Max(res1, res2), one/ulp)

	return
}
