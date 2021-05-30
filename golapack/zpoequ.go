package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zpoequ computes row and column scalings intended to equilibrate a
// Hermitian positive definite matrix A and reduce its condition number
// (with respect to the two-norm).  S contains the scale factors,
// S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
// elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
// choice of S puts the condition number of B within a factor N of the
// smallest possible condition number over all possible diagonal
// scalings.
func Zpoequ(n *int, a *mat.CMatrix, lda *int, s *mat.Vector, scond, amax *float64, info *int) {
	var one, smin, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*lda) < maxint(1, *n) {
		(*info) = -3
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPOEQU"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		(*scond) = one
		(*amax) = zero
		return
	}

	//     Find the minimum and maximum diagonal elements.
	s.Set(0, real(a.Get(0, 0)))
	smin = s.Get(0)
	(*amax) = s.Get(0)
	for i = 2; i <= (*n); i++ {
		s.Set(i-1, real(a.Get(i-1, i-1)))
		smin = minf64(smin, s.Get(i-1))
		(*amax) = maxf64(*amax, s.Get(i-1))
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
