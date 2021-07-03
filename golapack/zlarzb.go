package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlarzb applies a complex block reflector H or its transpose H**H
// to a complex distributed M-by-N  C from the left or the right.
//
// Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
func Zlarzb(side, trans, direct, storev byte, m, n, k, l *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CMatrix, ldwork *int) {
	var transt byte
	var one complex128
	var i, info, j int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZLARZB"), -info)
		return
	}

	if trans == 'N' {
		transt = 'C'
	} else {
		transt = 'N'
	}

	if side == 'L' {
		//        Form  H * C  or  H**H * C
		//
		//        W( 1:n, 1:k ) = C( 1:k, 1:n )**H
		for j = 1; j <= (*k); j++ {
			goblas.Zcopy(*n, c.CVector(j-1, 0), *ldc, work.CVector(0, j-1), 1)
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) + ...
		//                        C( m-l+1:m, 1:n )**H * V( 1:k, 1:l )**T
		if (*l) > 0 {
			err = goblas.Zgemm(Trans, ConjTrans, *n, *k, *l, one, c.Off((*m)-(*l)+1-1, 0), *ldc, v, *ldv, one, work, *ldwork)
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) * T**T  or  W( 1:m, 1:k ) * T
		err = goblas.Ztrmm(Right, Lower, mat.TransByte(transt), NonUnit, *n, *k, one, t, *ldt, work, *ldwork)

		//        C( 1:k, 1:n ) = C( 1:k, 1:n ) - W( 1:n, 1:k )**H
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(j-1, i-1))
			}
		}

		//        C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
		//                            V( 1:k, 1:l )**H * W( 1:n, 1:k )**H
		if (*l) > 0 {
			err = goblas.Zgemm(Trans, Trans, *l, *n, *k, -one, v, *ldv, work, *ldwork, one, c.Off((*m)-(*l)+1-1, 0), *ldc)
		}

	} else if side == 'R' {
		//        Form  C * H  or  C * H**H
		//
		//        W( 1:m, 1:k ) = C( 1:m, 1:k )
		for j = 1; j <= (*k); j++ {
			goblas.Zcopy(*m, c.CVector(0, j-1), 1, work.CVector(0, j-1), 1)
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) + ...
		//                        C( 1:m, n-l+1:n ) * V( 1:k, 1:l )**H
		if (*l) > 0 {
			err = goblas.Zgemm(NoTrans, Trans, *m, *k, *l, one, c.Off(0, (*n)-(*l)+1-1), *ldc, v, *ldv, one, work, *ldwork)
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) * conjg( T )  or
		//                        W( 1:m, 1:k ) * T**H
		for j = 1; j <= (*k); j++ {
			Zlacgv(toPtr((*k)-j+1), t.CVector(j-1, j-1), func() *int { y := 1; return &y }())
		}
		err = goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)
		for j = 1; j <= (*k); j++ {
			Zlacgv(toPtr((*k)-j+1), t.CVector(j-1, j-1), func() *int { y := 1; return &y }())
		}

		//        C( 1:m, 1:k ) = C( 1:m, 1:k ) - W( 1:m, 1:k )
		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		//        C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
		//                            W( 1:m, 1:k ) * conjg( V( 1:k, 1:l ) )
		for j = 1; j <= (*l); j++ {
			Zlacgv(k, v.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
		if (*l) > 0 {
			err = goblas.Zgemm(NoTrans, NoTrans, *m, *l, *k, -one, work, *ldwork, v, *ldv, one, c.Off(0, (*n)-(*l)+1-1), *ldc)
		}
		for j = 1; j <= (*l); j++ {
			Zlacgv(k, v.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	}
}
