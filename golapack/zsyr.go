package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsyr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a complex scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func Zsyr(uplo byte, n *int, alpha *complex128, x *mat.CVector, incx *int, a *mat.CMatrix, lda *int) {
	var temp, zero complex128
	var i, info, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != 'U' && uplo != 'L' {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*lda) < maxint(1, *n) {
		info = 7
	}
	if info != 0 {
		gltest.Xerbla([]byte("ZSYR  "), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || ((*alpha) == zero) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == 'U' {
		//        Form  A  when A is stored in upper triangle.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = (*alpha) * x.Get(j-1)
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
				}
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = (*alpha) * x.Get(jx-1)
					ix = kx
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
						ix = ix + (*incx)
					}
				}
				jx = jx + (*incx)
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = (*alpha) * x.Get(j-1)
					for i = j; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
				}
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = (*alpha) * x.Get(jx-1)
					ix = jx
					for i = j; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
						ix = ix + (*incx)
					}
				}
				jx = jx + (*incx)
			}
		}
	}
}
