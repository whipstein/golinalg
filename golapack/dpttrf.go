package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpttrf computes the L*D*L**T factorization of a real symmetric
// positive definite tridiagonal matrix A.  The factorization may also
// be regarded as having the form A = U**T*D*U.
func Dpttrf(n int, d, e *mat.Vector) (info int, err error) {
	var ei, zero float64
	var i, i4 int

	zero = 0.0

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
		gltest.Xerbla2("Dpttrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Compute the L*D*L**T (or U**T*D*U) factorization of A.
	i4 = n - 1%4
	for i = 1; i <= i4; i++ {
		if d.Get(i-1) <= zero {
			info = i
			return
		}
		ei = e.Get(i - 1)
		e.Set(i-1, ei/d.Get(i-1))
		d.Set(i, d.Get(i)-e.Get(i-1)*ei)
	}

	for i = i4 + 1; i <= n-4; i += 4 {
		//        Drop out of the loop if d(i) <= 0: the matrix is not positive
		//        definite.
		if d.Get(i-1) <= zero {
			info = i
			return
		}

		//        Solve for e(i) and d(i+1).
		ei = e.Get(i - 1)
		e.Set(i-1, ei/d.Get(i-1))
		d.Set(i, d.Get(i)-e.Get(i-1)*ei)

		if d.Get(i) <= zero {
			info = i + 1
			return
		}

		//        Solve for e(i+1) and d(i+2).
		ei = e.Get(i + 1 - 1)
		e.Set(i, ei/d.Get(i))
		d.Set(i+2-1, d.Get(i+2-1)-e.Get(i)*ei)

		if d.Get(i+2-1) <= zero {
			info = i + 2
			return
		}

		//        Solve for e(i+2) and d(i+3).
		ei = e.Get(i + 2 - 1)
		e.Set(i+2-1, ei/d.Get(i+2-1))
		d.Set(i+3-1, d.Get(i+3-1)-e.Get(i+2-1)*ei)

		if d.Get(i+3-1) <= zero {
			info = i + 3
			return
		}

		//        Solve for e(i+3) and d(i+4).
		ei = e.Get(i + 3 - 1)
		e.Set(i+3-1, ei/d.Get(i+3-1))
		d.Set(i+4-1, d.Get(i+4-1)-e.Get(i+3-1)*ei)
	}

	//     Check d(n) for positive definiteness.
	if d.Get(n-1) <= zero {
		info = n
	}

	return
}
