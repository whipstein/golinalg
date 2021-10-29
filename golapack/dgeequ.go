package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeequ computes row and column scalings intended to equilibrate an
// M-by-N matrix A and reduce its condition number.  R returns the row
// scale factors and C the column scale factors, chosen to try to make
// the largest element in each row and column of the matrix B with
// elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.
//
// R(i) and C(j) are restricted to be between SMLNUM = smallest safe
// number and BIGNUM = largest safe number.  Use of these scaling
// factors is not guaranteed to reduce the condition number of A but
// works well in practice.
func Dgeequ(m, n int, a *mat.Matrix, r, c *mat.Vector) (rowcnd, colcnd, amax float64, info int, err error) {
	var bignum, one, rcmax, rcmin, smlnum, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if m < 0 {
		info = -1
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		info = -2
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		info = -4
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgeequ", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		rowcnd = one
		colcnd = one
		amax = zero
		return
	}

	//     Get machine constants.
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum

	//     Compute row scale factors.
	for i = 1; i <= m; i++ {
		r.Set(i-1, zero)
	}

	//     Find the maximum element in each row.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			r.Set(i-1, math.Max(r.Get(i-1), math.Abs(a.Get(i-1, j-1))))
		}
	}

	//     Find the maximum and minimum scale factors.
	rcmin = bignum
	rcmax = zero
	for i = 1; i <= m; i++ {
		rcmax = math.Max(rcmax, r.Get(i-1))
		rcmin = math.Min(rcmin, r.Get(i-1))
	}
	amax = rcmax

	if rcmin == zero {
		//        Find the first zero scale factor and return an error code.
		for i = 1; i <= m; i++ {
			if r.Get(i-1) == zero {
				info = i
				return
			}
		}
	} else {
		//        Invert the scale factors.
		for i = 1; i <= m; i++ {
			r.Set(i-1, one/math.Min(math.Max(r.Get(i-1), smlnum), bignum))
		}

		//        Compute ROWCND = min(R(I)) / max(R(I))
		rowcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
	}

	//     Compute column scale factors
	for j = 1; j <= n; j++ {
		c.Set(j-1, zero)
	}

	//     Find the maximum element in each column,
	//     assuming the row scaling computed above.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			c.Set(j-1, math.Max(c.Get(j-1), math.Abs(a.Get(i-1, j-1))*r.Get(i-1)))
		}
	}

	//     Find the maximum and minimum scale factors.
	rcmin = bignum
	rcmax = zero
	for j = 1; j <= n; j++ {
		rcmin = math.Min(rcmin, c.Get(j-1))
		rcmax = math.Max(rcmax, c.Get(j-1))
	}

	if rcmin == zero {
		//        Find the first zero scale factor and return an error code.
		for j = 1; j <= n; j++ {
			if c.Get(j-1) == zero {
				info = m + j
				return
			}
		}
	} else {
		//        Invert the scale factors.
		for j = 1; j <= n; j++ {
			c.Set(j-1, one/math.Min(math.Max(c.Get(j-1), smlnum), bignum))
		}

		//        Compute COLCND = min(C(J)) / max(C(J))
		colcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
	}

	return
}
