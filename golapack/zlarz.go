package golapack

import (
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
func Zlarz(side mat.MatSide, m, n, l int, v *mat.CVector, incv int, tau complex128, c *mat.CMatrix, work *mat.CVector) {
	var one, zero complex128
	var err error

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	if side == Left {
		//        Form  H * C
		if tau != zero {
			//           w( 1:n ) = conjg( C( 1, 1:n ) )
			work.Copy(n, c.Off(0, 0).CVector(), c.Rows, 1)
			Zlacgv(n, work, 1)

			//           w( 1:n ) = conjg( w( 1:n ) + C( m-l+1:m, 1:n )**H * v( 1:l ) )
			err = work.Gemv(ConjTrans, l, n, one, c.Off(m-l, 0), v, incv, one, 1)
			Zlacgv(n, work, 1)

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			c.Off(0, 0).CVector().Axpy(n, -tau, work, 1, c.Rows)

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**H
			if err = c.Off(m-l, 0).Geru(l, n, -tau, v, incv, work, 1); err != nil {
				panic(err)
			}
		}

	} else {
		//        Form  C * H
		if tau != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			work.Copy(m, c.Off(0, 0).CVector(), 1, 1)

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			if err = work.Gemv(NoTrans, m, l, one, c.Off(0, n-l), v, incv, one, 1); err != nil {
				panic(err)
			}

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			c.Off(0, 0).CVector().Axpy(m, -tau, work, 1, 1)

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**H
			if err = c.Off(0, n-l).Gerc(m, l, -tau, work, 1, v, incv); err != nil {
				panic(err)
			}

		}

	}
}
