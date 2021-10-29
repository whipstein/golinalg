package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsyr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a complex scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func Zsyr(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, a *mat.CMatrix) (err error) {
	var temp, zero complex128
	var i, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if x.Inc == 0 {
		err = fmt.Errorf("x.Inc == 0: x.Inc=%v", x.Inc)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zsyr", err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	if x.Inc <= 0 {
		kx = 1 - (n-1)*x.Inc
	} else if x.Inc != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in upper triangle.
		if x.Inc == 1 {
			for j = 1; j <= n; j++ {
				if x.Get(j-1) != zero {
					temp = alpha * x.Get(j-1)
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
				}
			}
		} else {
			jx = kx
			for j = 1; j <= n; j++ {
				if x.Get(jx-1) != zero {
					temp = alpha * x.Get(jx-1)
					ix = kx
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
						ix = ix + x.Inc
					}
				}
				jx = jx + x.Inc
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		if x.Inc == 1 {
			for j = 1; j <= n; j++ {
				if x.Get(j-1) != zero {
					temp = alpha * x.Get(j-1)
					for i = j; i <= n; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
				}
			}
		} else {
			jx = kx
			for j = 1; j <= n; j++ {
				if x.Get(jx-1) != zero {
					temp = alpha * x.Get(jx-1)
					ix = jx
					for i = j; i <= n; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
						ix = ix + x.Inc
					}
				}
				jx = jx + x.Inc
			}
		}
	}

	return
}
