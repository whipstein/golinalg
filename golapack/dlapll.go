package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlapll Given two column vectors X and Y, let
//
//                      A = ( X Y ).
//
// The subroutine first computes the QR factorization of A = Q*R,
// and then computes the SVD of the 2-by-2 upper triangular matrix R.
// The smaller singular value of R is returned in SSMIN, which is used
// as the measurement of the linear dependency of the vectors X and Y.
func Dlapll(n int, x *mat.Vector, y *mat.Vector) (ssmin float64) {
	var a11, a12, a22, c, one, tau, zero float64

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if n <= 1 {
		ssmin = zero
		return
	}

	//     Compute the QR factorization of the N-by-2 matrix ( X Y )
	*x.GetPtr(0), tau = Dlarfg(n, x.Get(0), x.Off(1+x.Inc-1))
	a11 = x.Get(0)
	x.Set(0, one)

	c = -tau * goblas.Ddot(n, x, y)
	goblas.Daxpy(n, c, x, y)

	*y.GetPtr(1 + y.Inc - 1), tau = Dlarfg(n-1, y.Get(1+y.Inc-1), y.Off(1+2*y.Inc-1))

	a12 = y.Get(0)
	a22 = y.Get(1 + y.Inc - 1)

	//     Compute the SVD of 2-by-2 Upper triangular matrix.
	ssmin, _ = Dlas2(a11, a12, a22)

	return
}
