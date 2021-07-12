package golapack

import (
	"github.com/whipstein/golinalg/goblas"
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
func Dlarz(side byte, m, n, l *int, v *mat.Matrix, incv *int, tau *float64, c *mat.Matrix, ldc *int, work *mat.Vector) {
	var one, zero float64
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	if side == 'L' {
		//        Form  H * C
		if (*tau) != zero {
			//           w( 1:n ) = C( 1, 1:n )
			goblas.Dcopy(*n, c.VectorIdx(0), work.Off(0, 1))

			//           w( 1:n ) = w( 1:n ) + C( m-l+1:m, 1:n )**T * v( 1:l )
			err = goblas.Dgemv(Trans, *l, *n, one, c.Off((*m)-(*l), 0), v.VectorIdx(0), one, work.Off(0, 1))

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			goblas.Daxpy(*n, -(*tau), work.Off(0, 1), c.VectorIdx(0))

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**T
			err = goblas.Dger(*l, *n, -(*tau), v.VectorIdx(0), work.Off(0, 1), c.Off((*m)-(*l), 0))
		}

	} else {
		//        Form  C * H
		if (*tau) != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			goblas.Dcopy(*m, c.VectorIdx(0, 1), work.Off(0, 1))

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			err = goblas.Dgemv(NoTrans, *m, *l, one, c.Off(0, (*n)-(*l)), v.VectorIdx(0), one, work.Off(0, 1))

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			goblas.Daxpy(*m, -(*tau), work.Off(0, 1), c.VectorIdx(0, 1))

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**T
			err = goblas.Dger(*m, *l, -(*tau), work.Off(0, 1), v.VectorIdx(0), c.Off(0, (*n)-(*l)))

		}

	}
}
