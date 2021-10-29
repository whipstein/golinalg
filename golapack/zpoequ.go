package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpoequ computes row and column scalings intended to equilibrate a
// Hermitian positive definite matrix A and reduce its condition number
// (with respect to the two-norm).  S contains the scale factors,
// S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
// elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
// choice of S puts the condition number of B within a factor N of the
// smallest possible condition number over all possible diagonal
// scalings.
func Zpoequ(n int, a *mat.CMatrix, s *mat.Vector) (scond, amax float64, info int, err error) {
	var one, smin, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpoequ", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		scond = one
		amax = zero
		return
	}

	//     Find the minimum and maximum diagonal elements.
	s.Set(0, real(a.Get(0, 0)))
	smin = s.Get(0)
	amax = s.Get(0)
	for i = 2; i <= n; i++ {
		s.Set(i-1, real(a.Get(i-1, i-1)))
		smin = math.Min(smin, s.Get(i-1))
		amax = math.Max(amax, s.Get(i-1))
	}

	if smin <= zero {
		//        Find the first non-positive diagonal element and return.
		for i = 1; i <= n; i++ {
			if s.Get(i-1) <= zero {
				info = i
				return
			}
		}
	} else {
		//        Set the scale factors to the reciprocals
		//        of the diagonal elements.
		for i = 1; i <= n; i++ {
			s.Set(i-1, one/math.Sqrt(s.Get(i-1)))
		}

		//        Compute SCOND = min(S(I)) / max(S(I))
		scond = math.Sqrt(smin) / math.Sqrt(amax)
	}

	return
}
