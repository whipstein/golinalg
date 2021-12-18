package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarzb applies a real block reflector H or its transpose H**T to
// a real distributed M-by-N  C from the left or the right.
//
// Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
func Dlarzb(side mat.MatSide, trans mat.MatTrans, direct, storev byte, m, n, k, l int, v, t, c, work *mat.Matrix) (err error) {
	var transt mat.MatTrans
	var one float64
	var i, j int

	one = 1.0

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
		gltest.Xerbla2("Dlarzb", err)
		return
	}

	if trans == NoTrans {
		transt = Trans
	} else {
		transt = NoTrans
	}

	if side == Left {
		//        Form  H * C  or  H**T * C
		//
		//        W( 1:n, 1:k ) = C( 1:k, 1:n )**T
		for j = 1; j <= k; j++ {
			work.Off(0, j-1).Vector().Copy(n, c.Off(j-1, 0).Vector(), c.Rows, 1)
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) + ...
		//                        C( m-l+1:m, 1:n )**T * V( 1:k, 1:l )**T
		if l > 0 {
			if err = work.Gemm(Trans, Trans, n, k, l, one, c.Off(m-l, 0), v, one); err != nil {
				panic(err)
			}
		}

		//        W( 1:n, 1:k ) = W( 1:n, 1:k ) * T**T  or  W( 1:m, 1:k ) * T
		if err = work.Trmm(Right, Lower, transt, NonUnit, n, k, one, t); err != nil {
			panic(err)
		}

		//        C( 1:k, 1:n ) = C( 1:k, 1:n ) - W( 1:n, 1:k )**T
		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(j-1, i-1))
			}
		}

		//        C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
		//                            V( 1:k, 1:l )**T * W( 1:n, 1:k )**T
		if l > 0 {
			if err = c.Off(m-l, 0).Gemm(Trans, Trans, l, n, k, -one, v, work, one); err != nil {
				panic(err)
			}
		}

	} else if side == Right {
		//        Form  C * H  or  C * H**T
		//
		//        W( 1:m, 1:k ) = C( 1:m, 1:k )
		for j = 1; j <= k; j++ {
			work.Off(0, j-1).Vector().Copy(m, c.Off(0, j-1).Vector(), 1, 1)
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) + ...
		//                        C( 1:m, n-l+1:n ) * V( 1:k, 1:l )**T
		if l > 0 {
			if err = work.Gemm(NoTrans, Trans, m, k, l, one, c.Off(0, n-l), v, one); err != nil {
				panic(err)
			}
		}

		//        W( 1:m, 1:k ) = W( 1:m, 1:k ) * T  or  W( 1:m, 1:k ) * T**T
		if err = work.Trmm(Right, Lower, trans, NonUnit, m, k, one, t); err != nil {
			panic(err)
		}

		//        C( 1:m, 1:k ) = C( 1:m, 1:k ) - W( 1:m, 1:k )
		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		//        C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
		//                            W( 1:m, 1:k ) * V( 1:k, 1:l )
		if l > 0 {
			if err = c.Off(0, n-l).Gemm(NoTrans, NoTrans, m, l, k, -one, work, v, one); err != nil {
				panic(err)
			}
		}

	}

	return
}
