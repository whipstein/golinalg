package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlapll Given two column vectors X and Y, let
//
//                      A = ( X Y ).
//
// The subroutine first computes the QR factorization of A = Q*R,
// and then computes the SVD of the 2-by-2 upper triangular matrix R.
// The smaller singular value of R is returned in SSMIN, which is used
// as the measurement of the linear dependency of the vectors X and Y.
func Zlapll(n int, x *mat.CVector, incx int, y *mat.CVector, incy int) (ssmin float64) {
	var a11, a12, a22, c, cone, tau complex128
	var zero float64

	zero = 0.0
	cone = (1.0 + 0.0*1i)

	//     Quick return if possible
	if n <= 1 {
		ssmin = zero
		return
	}

	//     Compute the QR factorization of the N-by-2 matrix ( X Y )
	*x.GetPtr(0), tau = Zlarfg(n, x.Get(0), x.Off(1+incx-1), incx)
	a11 = x.Get(0)
	x.Set(0, cone)

	c = -cmplx.Conj(tau) * y.Dotc(n, x, incx, incy)
	y.Axpy(n, c, x, incx, incy)

	*y.GetPtr(1 + incy - 1), tau = Zlarfg(n-1, y.Get(1+incy-1), y.Off(1+2*incy-1), incy)

	a12 = y.Get(0)
	a22 = y.Get(1 + incy - 1)

	//     Compute the SVD of 2-by-2 Upper triangular matrix.
	ssmin, _ = Dlas2(cmplx.Abs(a11), cmplx.Abs(a12), cmplx.Abs(a22))

	return
}
