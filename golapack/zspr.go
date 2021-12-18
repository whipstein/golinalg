package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zspr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a complex scalar, x is an n element vector and A is an
// n by n symmetric matrix, supplied in packed form.
func Zspr(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, incx int, ap *mat.CVector) (err error) {
	var temp, zero complex128
	var i, ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx == 0: incx=%v", incx)
	}
	if err != nil {
		gltest.Xerbla2("Zspr", err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		if incx == 1 {
			for j = 1; j <= n; j++ {
				if x.Get(j-1) != zero {
					temp = alpha * x.Get(j-1)
					k = kk
					for i = 1; i <= j-1; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k = k + 1
					}
					ap.Set(kk+j-1-1, ap.Get(kk+j-1-1)+x.Get(j-1)*temp)
				} else {
					ap.Set(kk+j-1-1, ap.Get(kk+j-1-1))
				}
				kk = kk + j
			}
		} else {
			jx = kx
			for j = 1; j <= n; j++ {
				if x.Get(jx-1) != zero {
					temp = alpha * x.Get(jx-1)
					ix = kx
					for k = kk; k <= kk+j-2; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
						ix = ix + incx
					}
					ap.Set(kk+j-1-1, ap.Get(kk+j-1-1)+x.Get(jx-1)*temp)
				} else {
					ap.Set(kk+j-1-1, ap.Get(kk+j-1-1))
				}
				jx = jx + incx
				kk = kk + j
			}
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		if incx == 1 {
			for j = 1; j <= n; j++ {
				if x.Get(j-1) != zero {
					temp = alpha * x.Get(j-1)
					ap.Set(kk-1, ap.Get(kk-1)+temp*x.Get(j-1))
					k = kk + 1
					for i = j + 1; i <= n; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k = k + 1
					}
				} else {
					ap.Set(kk-1, ap.Get(kk-1))
				}
				kk = kk + n - j + 1
			}
		} else {
			jx = kx
			for j = 1; j <= n; j++ {
				if x.Get(jx-1) != zero {
					temp = alpha * x.Get(jx-1)
					ap.Set(kk-1, ap.Get(kk-1)+temp*x.Get(jx-1))
					ix = jx
					for k = kk + 1; k <= kk+n-j; k++ {
						ix = ix + incx
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
					}
				} else {
					ap.Set(kk-1, ap.Get(kk-1))
				}
				jx = jx + incx
				kk = kk + n - j + 1
			}
		}
	}

	return
}
