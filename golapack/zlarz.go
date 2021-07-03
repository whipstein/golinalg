package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlarz applies a complex elementary reflector H to a complex
// M-by-N matrix C, from either the left or the right. H is represented
// in the form
//
//       H = I - tau * v * v**H
//
// where tau is a complex scalar and v is a complex vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
// To apply H**H (the conjugate transpose of H), supply conjg(tau) instead
// tau.
//
// H is a product of k elementary reflectors as returned by ZTZRZF.
func Zlarz(side byte, m, n, l *int, v *mat.CVector, incv *int, tau *complex128, c *mat.CMatrix, ldc *int, work *mat.CVector) {
	var one, zero complex128
	var err error
	_ = err

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	if side == 'L' {
		//        Form  H * C
		if (*tau) != zero {
			//           w( 1:n ) = conjg( C( 1, 1:n ) )
			goblas.Zcopy(*n, c.CVector(0, 0), *ldc, work, 1)
			Zlacgv(n, work, func() *int { y := 1; return &y }())

			//           w( 1:n ) = conjg( w( 1:n ) + C( m-l+1:m, 1:n )**H * v( 1:l ) )
			err = goblas.Zgemv(ConjTrans, *l, *n, one, c.Off((*m)-(*l)+1-1, 0), *ldc, v, *incv, one, work, 1)
			Zlacgv(n, work, func() *int { y := 1; return &y }())

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			goblas.Zaxpy(*n, -(*tau), work, 1, c.CVector(0, 0), *ldc)

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**H
			err = goblas.Zgeru(*l, *n, -(*tau), v, *incv, work, 1, c.Off((*m)-(*l)+1-1, 0), *ldc)
		}

	} else {
		//        Form  C * H
		if (*tau) != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			goblas.Zcopy(*m, c.CVector(0, 0), 1, work, 1)

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			err = goblas.Zgemv(NoTrans, *m, *l, one, c.Off(0, (*n)-(*l)+1-1), *ldc, v, *incv, one, work, 1)

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			goblas.Zaxpy(*m, -(*tau), work, 1, c.CVector(0, 0), 1)

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**H
			err = goblas.Zgerc(*m, *l, -(*tau), work, 1, v, *incv, c.Off(0, (*n)-(*l)+1-1), *ldc)

		}

	}
}
