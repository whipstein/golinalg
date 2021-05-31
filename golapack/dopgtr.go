package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dopgtr generates a real orthogonal matrix Q which is defined as the
// product of n-1 elementary reflectors H(i) of order n, as returned by
// DSPTRD using packed storage:
//
// if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
func Dopgtr(uplo byte, n *int, ap, tau *mat.Vector, q *mat.Matrix, ldq *int, work *mat.Vector, info *int) {
	var upper bool
	var one, zero float64
	var i, iinfo, ij, j int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldq) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DOPGTR"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Q was determined by a call to DSPTRD with UPLO = 'U'
		//
		//        Unpack the vectors which define the elementary reflectors and
		//        set the last row and column of Q equal to those of the unit
		//        matrix
		ij = 2
		for j = 1; j <= (*n)-1; j++ {
			for i = 1; i <= j-1; i++ {
				q.Set(i-1, j-1, ap.Get(ij-1))
				ij = ij + 1
			}
			ij = ij + 2
			q.Set((*n)-1, j-1, zero)
		}
		for i = 1; i <= (*n)-1; i++ {
			q.Set(i-1, (*n)-1, zero)
		}
		q.Set((*n)-1, (*n)-1, one)

		//        Generate Q(1:n-1,1:n-1)
		Dorg2l(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), q, ldq, tau, work, &iinfo)

	} else {
		//        Q was determined by a call to DSPTRD with UPLO = 'L'.
		//
		//        Unpack the vectors which define the elementary reflectors and
		//        set the first row and column of Q equal to those of the unit
		//        matrix
		q.Set(0, 0, one)
		for i = 2; i <= (*n); i++ {
			q.Set(i-1, 0, zero)
		}
		ij = 3
		for j = 2; j <= (*n); j++ {
			q.Set(0, j-1, zero)
			for i = j + 1; i <= (*n); i++ {
				q.Set(i-1, j-1, ap.Get(ij-1))
				ij = ij + 1
			}
			ij = ij + 2
		}
		if (*n) > 1 {
			//           Generate Q(2:n,2:n)
			Dorg2r(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), q.Off(1, 1), ldq, tau, work, &iinfo)
		}
	}
}
