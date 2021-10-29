package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpbequ computes row and column scalings intended to equilibrate a
// Hermitian positive definite band matrix A and reduce its condition
// number (with respect to the two-norm).  S contains the scale factors,
// S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
// elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
// choice of S puts the condition number of B within a factor N of the
// smallest possible condition number over all possible diagonal
// scalings.
func Zpbequ(uplo mat.MatUplo, n, kd int, ab *mat.CMatrix, s *mat.Vector) (scond, amax float64, info int, err error) {
	var upper bool
	var one, smin, zero float64
	var i, j int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	}
	if err != nil {
		gltest.Xerbla2("Zpbequ", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		scond = one
		amax = zero
		return
	}

	if upper {
		j = kd + 1
	} else {
		j = 1
	}

	//     Initialize SMIN and AMAX.
	s.Set(0, ab.GetRe(j-1, 0))
	smin = s.Get(0)
	amax = s.Get(0)

	//     Find the minimum and maximum diagonal elements.
	for i = 2; i <= n; i++ {
		s.Set(i-1, ab.GetRe(j-1, i-1))
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
