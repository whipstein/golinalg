package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsymv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func Zsymv(uplo byte, n *int, alpha *complex128, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var one, temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != 'U' && uplo != 'L' {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*lda) < maxint(1, *n) {
		info = 5
	} else if (*incx) == 0 {
		info = 7
	} else if (*incy) == 0 {
		info = 10
	}
	if info != 0 {
		gltest.Xerbla([]byte("ZSYMV "), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || (((*alpha) == zero) && ((*beta) == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	if (*incx) > 0 {
		kx = 1
	} else {
		kx = 1 - ((*n)-1)*(*incx)
	}
	if (*incy) > 0 {
		ky = 1
	} else {
		ky = 1 - ((*n)-1)*(*incy)
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if (*beta) != one {
		if (*incy) == 1 {
			if (*beta) == zero {
				for i = 1; i <= (*n); i++ {
					y.Set(i-1, zero)
				}
			} else {
				for i = 1; i <= (*n); i++ {
					y.Set(i-1, (*beta)*y.Get(i-1))
				}
			}
		} else {
			iy = ky
			if (*beta) == zero {
				for i = 1; i <= (*n); i++ {
					y.Set(iy-1, zero)
					iy = iy + (*incy)
				}
			} else {
				for i = 1; i <= (*n); i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy = iy + (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	if uplo == 'U' {
		//        Form  y  when A is stored in upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(j-1, j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				ix = kx
				iy = ky
				for i = 1; i <= j-1; i++ {
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(ix-1)
					ix = ix + (*incx)
					iy = iy + (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(j-1, j-1))
				for i = j + 1; i <= (*n); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1))
				ix = jx
				iy = jy
				for i = j + 1; i <= (*n); i++ {
					ix = ix + (*incx)
					iy = iy + (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	}
}
