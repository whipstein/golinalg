package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dlarz applies a real elementary reflector H to a real M-by-N
// matrix C, from either the left or the right. H is represented in the
// form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
//
// H is a product of k elementary reflectors as returned by DTZRZF.
func Dlarz(side mat.MatSide, m, n, l int, v *mat.Matrix, tau float64, c *mat.Matrix, work *mat.Vector) {
	var one, zero float64
	var err error

	one = 1.0
	zero = 0.0

	if side == Left {
		//        Form  H * C
		if tau != zero {
			//           w( 1:n ) = C( 1, 1:n )
			work.Copy(n, c.OffIdx(0).Vector(), c.Rows, 1)

			//           w( 1:n ) = w( 1:n ) + C( m-l+1:m, 1:n )**T * v( 1:l )
			if err = work.Gemv(Trans, l, n, one, c.Off(m-l, 0), v.OffIdx(0).Vector(), v.Rows, one, 1); err != nil {
				panic(err)
			}

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			c.OffIdx(0).Vector().Axpy(n, -tau, work, 1, c.Rows)

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**T
			if err = c.Off(m-l, 0).Ger(l, n, -tau, v.OffIdx(0).Vector(), v.Rows, work, 1); err != nil {
				panic(err)
			}
		}

	} else {
		//        Form  C * H
		if tau != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			work.Copy(m, c.OffIdx(0).Vector(), 1, 1)

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			if err = work.Gemv(NoTrans, m, l, one, c.Off(0, n-l), v.OffIdx(0).Vector(), v.Rows, one, 1); err != nil {
				panic(err)
			}

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			c.OffIdx(0).Vector().Axpy(m, -tau, work, 1, 1)

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**T
			if err = c.Off(0, n-l).Ger(m, l, -tau, work, 1, v.OffIdx(0).Vector(), v.Rows); err != nil {
				panic(err)
			}

		}

	}
}
