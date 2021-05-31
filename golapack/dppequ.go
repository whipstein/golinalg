package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dppequ computes row and column scalings intended to equilibrate a
// symmetric positive definite matrix A in packed storage and reduce
// its condition number (with respect to the two-norm).  S contains the
// scale factors, S(i)=1/sqrt(A(i,i)), chosen so that the scaled matrix
// B with elements B(i,j)=S(i)*A(i,j)*S(j) has ones on the diagonal.
// This choice of S puts the condition number of B within a factor N of
// the smallest possible condition number over all possible diagonal
// scalings.
func Dppequ(uplo byte, n *int, ap, s *mat.Vector, scond, amax *float64, info *int) {
	var upper bool
	var one, smin, zero float64
	var i, jj int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPPEQU"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		(*scond) = one
		(*amax) = zero
		return
	}

	//     Initialize SMIN and AMAX.
	s.Set(0, ap.Get(0))
	smin = s.Get(0)
	(*amax) = s.Get(0)

	if upper {
		//        UPLO = 'U':  Upper triangle of A is stored.
		//        Find the minimum and maximum diagonal elements.
		jj = 1
		for i = 2; i <= (*n); i++ {
			jj = jj + i
			s.Set(i-1, ap.Get(jj-1))
			smin = minf64(smin, s.Get(i-1))
			(*amax) = maxf64(*amax, s.Get(i-1))
		}

	} else {
		//        UPLO = 'L':  Lower triangle of A is stored.
		//        Find the minimum and maximum diagonal elements.
		jj = 1
		for i = 2; i <= (*n); i++ {
			jj = jj + (*n) - i + 2
			s.Set(i-1, ap.Get(jj-1))
			smin = minf64(smin, s.Get(i-1))
			(*amax) = maxf64(*amax, s.Get(i-1))
		}
	}

	if smin <= zero {
		//        Find the first non-positive diagonal element and return.
		for i = 1; i <= (*n); i++ {
			if s.Get(i-1) <= zero {
				(*info) = i
				return
			}
		}
	} else {
		//        Set the scale factors to the reciprocals
		//        of the diagonal elements.
		for i = 1; i <= (*n); i++ {
			s.Set(i-1, one/math.Sqrt(s.Get(i-1)))
		}

		//        Compute SCOND = min(S(I)) / max(S(I))
		(*scond) = math.Sqrt(smin) / math.Sqrt(*amax)
	}
}
