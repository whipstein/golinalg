package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsymv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func Zsymv(uplo mat.MatUplo, n int, alpha complex128, a *mat.CMatrix, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var one, temp1, temp2, zero complex128
	var i, ix, iy, j, jx, jy, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	// Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if x.Inc == 0 {
		err = fmt.Errorf("x.Inc == 0: x.Inc=%v", x.Inc)
	} else if y.Inc == 0 {
		err = fmt.Errorf("y.Inc == 0: y.Inc=%v", y.Inc)
	}
	if err != nil {
		gltest.Xerbla2("Zsymv", err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	if x.Inc > 0 {
		kx = 1
	} else {
		kx = 1 - (n-1)*x.Inc
	}
	if y.Inc > 0 {
		ky = 1
	} else {
		ky = 1 - (n-1)*y.Inc
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if y.Inc == 1 {
			if beta == zero {
				for i = 1; i <= n; i++ {
					y.Set(i-1, zero)
				}
			} else {
				for i = 1; i <= n; i++ {
					y.Set(i-1, beta*y.Get(i-1))
				}
			}
		} else {
			iy = ky
			if beta == zero {
				for i = 1; i <= n; i++ {
					y.Set(iy-1, zero)
					iy = iy + y.Inc
				}
			} else {
				for i = 1; i <= n; i++ {
					y.Set(iy-1, beta*y.Get(iy-1))
					iy = iy + y.Inc
				}
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		if (x.Inc == 1) && (y.Inc == 1) {
			for j = 1; j <= n; j++ {
				temp1 = alpha * x.Get(j-1)
				temp2 = zero
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(j-1, j-1)+alpha*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= n; j++ {
				temp1 = alpha * x.Get(jx-1)
				temp2 = zero
				ix = kx
				iy = ky
				for i = 1; i <= j-1; i++ {
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(ix-1)
					ix = ix + x.Inc
					iy = iy + y.Inc
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1)+alpha*temp2)
				jx = jx + x.Inc
				jy = jy + y.Inc
			}
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		if (x.Inc == 1) && (y.Inc == 1) {
			for j = 1; j <= n; j++ {
				temp1 = alpha * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(j-1, j-1))
				for i = j + 1; i <= n; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+alpha*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= n; j++ {
				temp1 = alpha * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1))
				ix = jx
				iy = jy
				for i = j + 1; i <= n; i++ {
					ix = ix + x.Inc
					iy = iy + y.Inc
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.Get(i-1, j-1)*x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
				jx = jx + x.Inc
				jy = jy + y.Inc
			}
		}
	}

	return
}
