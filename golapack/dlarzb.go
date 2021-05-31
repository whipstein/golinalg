package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarzb applies a real block reflector H or its transpose H**T to
// a real distributed M-by-N  C from the left or the right.
//
// Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
func Dlarzb(side, trans, direct, storev byte, m, n, k, l *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, c *mat.Matrix, ldc *int, work *mat.Matrix, ldwork *int) {
	var transt byte
	var one float64
	var i, info, j int

	one = 1.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	//     Check for currently supported options
	info = 0
	if direct != 'B' {
		info = -3
	} else if storev != 'R' {
		info = -4
	}
	if info != 0 {
		gltest.Xerbla([]byte("DLARZB"), -info)
		return
	}

	if trans == 'N' {
		transt = 'T'
	} else {
		transt = 'N'
	}

	if side == 'L' {
		//        Form  H * C  or  H**T * C
		//
		//        W( 1:n, 1:k ) = C( 1:k, 1:n )**T
		for j = 1; j <= (*k); j++ {
			goblas.Dcopy(n, c.Vector(j-1, 0), ldc, work.Vector(0, j-1), toPtr(1))
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) + ...
		//                        C( m-l+1:m, 1:n )**T * V( 1:k, 1:l )**T
		if (*l) > 0 {
			goblas.Dgemm(Trans, Trans, n, k, l, &one, c.Off((*m)-(*l)+1-1, 0), ldc, v, ldv, &one, work, ldwork)
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) * T**T  or  W( 1:m, 1:k ) * T
		goblas.Dtrmm(Right, Lower, mat.TransByte(transt), NonUnit, n, k, &one, t, ldt, work, ldwork)

		//        C( 1:k, 1:n ) = C( 1:k, 1:n ) - W( 1:n, 1:k )**T
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(j-1, i-1))
			}
		}

		//        C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
		//                            V( 1:k, 1:l )**T * W( 1:n, 1:k )**T
		if (*l) > 0 {
			goblas.Dgemm(Trans, Trans, l, n, k, toPtrf64(-one), v, ldv, work, ldwork, &one, c.Off((*m)-(*l)+1-1, 0), ldc)
		}

	} else if side == 'R' {
		//        Form  C * H  or  C * H**T
		//
		//        W( 1:m, 1:k ) = C( 1:m, 1:k )
		for j = 1; j <= (*k); j++ {
			goblas.Dcopy(m, c.Vector(0, j-1), toPtr(1), work.Vector(0, j-1), toPtr(1))
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) + ...
		//                        C( 1:m, n-l+1:n ) * V( 1:k, 1:l )**T
		if (*l) > 0 {
			goblas.Dgemm(NoTrans, Trans, m, k, l, &one, c.Off(0, (*n)-(*l)+1-1), ldc, v, ldv, &one, work, ldwork)
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) * T  or  W( 1:m, 1:k ) * T**T
		goblas.Dtrmm(Right, Lower, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

		//        C( 1:m, 1:k ) = C( 1:m, 1:k ) - W( 1:m, 1:k )
		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		//        C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
		//                            W( 1:m, 1:k ) * V( 1:k, 1:l )
		if (*l) > 0 {
			goblas.Dgemm(NoTrans, NoTrans, m, l, k, toPtrf64(-one), work, ldwork, v, ldv, &one, c.Off(0, (*n)-(*l)+1-1), ldc)
		}

	}
}
