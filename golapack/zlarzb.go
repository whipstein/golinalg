package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlarzb applies a complex block reflector H or its transpose H**H
// to a complex distributed M-by-N  C from the left or the right.
//
// Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
func Zlarzb(side mat.MatSide, trans mat.MatTrans, direct, storev byte, m, n, k, l int, v, t, c, work *mat.CMatrix) (err error) {
	var transt mat.MatTrans
	var one complex128
	var i, j int

	one = (1.0 + 0.0*1i)

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	//     Check for currently supported options
	if direct != 'B' {
		err = fmt.Errorf("direct != 'B': direct='%c'", direct)
	} else if storev != 'R' {
		err = fmt.Errorf("storev != 'R': storev='%c'", storev)
	}
	if err != nil {
		gltest.Xerbla2("Zlarzb", err)
		return
	}

	if trans == NoTrans {
		transt = ConjTrans
	} else {
		transt = NoTrans
	}

	if side == Left {
		//        Form  H * C  or  H**H * C
		//
		//        W( 1:n, 1:k ) = C( 1:k, 1:n )**H
		for j = 1; j <= k; j++ {
			work.Off(0, j-1).CVector().Copy(n, c.Off(j-1, 0).CVector(), c.Rows, 1)
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) + ...
		//                        C( m-l+1:m, 1:n )**H * V( 1:k, 1:l )**T
		if l > 0 {
			if err = work.Gemm(Trans, ConjTrans, n, k, l, one, c.Off(m-l, 0), v, one); err != nil {
				panic(err)
			}
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) * T**T  or  W( 1:m, 1:k ) * T
		if err = work.Trmm(Right, Lower, transt, NonUnit, n, k, one, t); err != nil {
			panic(err)
		}

		//        C( 1:k, 1:n ) = C( 1:k, 1:n ) - W( 1:n, 1:k )**H
		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(j-1, i-1))
			}
		}

		//        C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
		//                            V( 1:k, 1:l )**H * W( 1:n, 1:k )**H
		if l > 0 {
			if err = c.Off(m-l, 0).Gemm(Trans, Trans, l, n, k, -one, v, work, one); err != nil {
				panic(err)
			}
		}

	} else if side == 'R' {
		//        Form  C * H  or  C * H**H
		//
		//        W( 1:m, 1:k ) = C( 1:m, 1:k )
		for j = 1; j <= k; j++ {
			work.Off(0, j-1).CVector().Copy(m, c.Off(0, j-1).CVector(), 1, 1)
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) + ...
		//                        C( 1:m, n-l+1:n ) * V( 1:k, 1:l )**H
		if l > 0 {
			if err = work.Gemm(NoTrans, Trans, m, k, l, one, c.Off(0, n-l), v, one); err != nil {
				panic(err)
			}
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) * conjg( T )  or
		//                        W( 1:m, 1:k ) * T**H
		for j = 1; j <= k; j++ {
			Zlacgv(k-j+1, t.Off(j-1, j-1).CVector(), 1)
		}
		if err = work.Trmm(Right, Lower, trans, NonUnit, m, k, one, t); err != nil {
			panic(err)
		}
		for j = 1; j <= k; j++ {
			Zlacgv(k-j+1, t.Off(j-1, j-1).CVector(), 1)
		}

		//        C( 1:m, 1:k ) = C( 1:m, 1:k ) - W( 1:m, 1:k )
		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		//        C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
		//                            W( 1:m, 1:k ) * conjg( V( 1:k, 1:l ) )
		for j = 1; j <= l; j++ {
			Zlacgv(k, v.Off(0, j-1).CVector(), 1)
		}
		if l > 0 {
			if err = c.Off(0, n-l).Gemm(NoTrans, NoTrans, m, l, k, -one, work, v, one); err != nil {
				panic(err)
			}
		}
		for j = 1; j <= l; j++ {
			Zlacgv(k, v.Off(0, j-1).CVector(), 1)
		}

	}

	return
}
