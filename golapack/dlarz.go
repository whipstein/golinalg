package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
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

	one = 1.0
	zero = 0.0

	if side == 'L' {
		//        Form  H * C
		if (*tau) != zero {
			//           w( 1:n ) = C( 1, 1:n )
			goblas.Dcopy(n, c.VectorIdx(0), ldc, work, toPtr(1))

			//           w( 1:n ) = w( 1:n ) + C( m-l+1:m, 1:n )**T * v( 1:l )
			goblas.Dgemv(Trans, l, n, &one, c.Off((*m)-(*l)+1-1, 0), ldc, v.VectorIdx(0), incv, &one, work, toPtr(1))

			//           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
			goblas.Daxpy(n, toPtrf64(-(*tau)), work, toPtr(1), c.VectorIdx(0), ldc)

			//           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
			//                               tau * v( 1:l ) * w( 1:n )**T
			goblas.Dger(l, n, toPtrf64(-(*tau)), v.VectorIdx(0), incv, work, toPtr(1), c.Off((*m)-(*l)+1-1, 0), ldc)
		}

	} else {
		//        Form  C * H
		if (*tau) != zero {
			//           w( 1:m ) = C( 1:m, 1 )
			goblas.Dcopy(m, c.VectorIdx(0), toPtr(1), work, toPtr(1))

			//           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
			goblas.Dgemv(NoTrans, m, l, &one, c.Off(0, (*n)-(*l)+1-1), ldc, v.VectorIdx(0), incv, &one, work, toPtr(1))

			//           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
			goblas.Daxpy(m, toPtrf64(-(*tau)), work, toPtr(1), c.VectorIdx(0), toPtr(1))

			//           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
			//                               tau * w( 1:m ) * v( 1:l )**T
			goblas.Dger(m, l, toPtrf64(-(*tau)), work, toPtr(1), v.VectorIdx(0), incv, c.Off(0, (*n)-(*l)+1-1), ldc)

		}

	}
}
