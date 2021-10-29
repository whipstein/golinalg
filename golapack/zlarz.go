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
func Zlarz(side mat.MatSide, m, n, l int, v *mat.CVector, tau complex128, c *mat.CMatrix, work *mat.CVector) {
	var one, zero complex128
	var err error

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	if side == Left {
		//        Form  H * C
		if tau != zero {
			//           w( 1:n ) = conjg( C( 1, 1:n ) )
			goblas.Zcopy(n, c.CVector(0, 0), work.Off(0, 1))
			Zlacgv(n, work.Off(0, 1))

			//           w( 1:n ) = conjg( w( 1:n ) + C( m-l+1:m, 1:n )**H * v( 1:l ) )
			err = goblas.Zgemv(ConjTrans, l, n, one, c.Off(m-l, 0), v, one, work.Off(0, 1))
			Zlacgv(n, work.Off(0, 1))

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			goblas.Zaxpy(n, -tau, work.Off(0, 1), c.CVector(0, 0))

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**H
			if err = goblas.Zgeru(l, n, -tau, v, work.Off(0, 1), c.Off(m-l, 0)); err != nil {
				panic(err)
			}
		}

	} else {
		//        Form  C * H
		if tau != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			goblas.Zcopy(m, c.CVector(0, 0, 1), work.Off(0, 1))

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			if err = goblas.Zgemv(NoTrans, m, l, one, c.Off(0, n-l), v, one, work.Off(0, 1)); err != nil {
				panic(err)
			}

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			goblas.Zaxpy(m, -tau, work.Off(0, 1), c.CVector(0, 0, 1))

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**H
			if err = goblas.Zgerc(m, l, -tau, work.Off(0, 1), v, c.Off(0, n-l)); err != nil {
				panic(err)
			}

		}

	}
}
