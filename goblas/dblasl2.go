package goblas

import (
	"golinalg/mat"
)

// Dgbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func Dgbmv(trans mat.MatTrans, m, n, kl, ku *int, alpha *float64, a *mat.Matrix, lda *int, x *mat.Vector, incx *int, beta *float64, y *mat.Vector, incy *int) {
	var one, temp, zero float64
	var i, info, ix, iy, j, jx, jy, k, kup1, kx, ky, lenx, leny int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !trans.IsValid() {
		info = 1
	} else if (*m) < 0 {
		info = 2
	} else if (*n) < 0 {
		info = 3
	} else if (*kl) < 0 {
		info = 4
	} else if (*ku) < 0 {
		info = 5
	} else if (*lda) < ((*kl) + (*ku) + 1) {
		info = 8
	} else if (*incx) == 0 {
		info = 10
	} else if (*incy) == 0 {
		info = 13
	}
	if info != 0 {
		Xerbla([]byte("Dgbmv"), info)
		return
	}

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) || (((*alpha) == zero) && ((*beta) == one)) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == mat.NoTrans {
		lenx = (*n)
		leny = (*m)
	} else {
		lenx = (*m)
		leny = (*n)
	}
	if (*incx) > 0 {
		kx = 1
	} else {
		kx = 1 - (lenx-1)*(*incx)
	}
	if (*incy) > 0 {
		ky = 1
	} else {
		ky = 1 - (leny-1)*(*incy)
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the band part of A.
	//
	//     First form  y := beta*y.
	if (*beta) != one {
		if (*incy) == 1 {
			if (*beta) == zero {
				for i = 1; i <= leny; i++ {
					y.Set(i-1, zero)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(i-1, (*beta)*y.Get(i-1))
				}
			}
		} else {
			iy = ky
			if (*beta) == zero {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, zero)
					iy += (*incy)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy += (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	kup1 = (*ku) + 1
	if trans == mat.NoTrans {
		//        Form  y := alpha*A*x + y.
		jx = kx
		if (*incy) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint((*m), j+(*kl)); i++ {
					y.Set(i-1, y.Get(i-1)+temp*a.Get(k+i-1, j-1))
				}
				jx += (*incx)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				iy = ky
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint((*m), j+(*kl)); i++ {
					y.Set(iy-1, y.Get(iy-1)+temp*a.Get(k+i-1, j-1))
					iy += (*incy)
				}
				jx += (*incx)
				if j > (*ku) {
					ky += (*incy)
				}
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y.
		jy = ky
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = zero
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint((*m), j+(*kl)); i++ {
					temp += a.Get(k+i-1, j-1) * x.Get(i-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy += (*incy)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = zero
				ix = kx
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint((*m), j+(*kl)); i++ {
					temp += a.Get(k+i-1, j-1) * x.Get(ix-1)
					ix += (*incx)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy += (*incy)
				if j > (*ku) {
					kx += (*incx)
				}
			}
		}
	}
}

// Dgemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func Dgemv(trans mat.MatTrans, m, n *int, alpha *float64, a *mat.Matrix, lda *int, x *mat.Vector, incx *int, beta *float64, y *mat.Vector, incy *int) {
	var one, temp, zero float64
	var i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !trans.IsValid() {
		info = 1
	} else if (*m) < 0 {
		info = 2
	} else if (*n) < 0 {
		info = 3
	} else if (*lda) < maxint(1, (*m)) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	} else if (*incy) == 0 {
		info = 11
	}
	if info != 0 {
		Xerbla([]byte("Dgemv"), info)
		return
	}

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) || (((*alpha) == zero) && ((*beta) == one)) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == mat.NoTrans {
		lenx = (*n)
		leny = (*m)
	} else {
		lenx = (*m)
		leny = (*n)
	}
	if (*incx) > 0 {
		kx = 1
	} else {
		kx = 1 - (lenx-1)*(*incx)
	}
	if (*incy) > 0 {
		ky = 1
	} else {
		ky = 1 - (leny-1)*(*incy)
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if (*beta) != one {
		if (*incy) == 1 {
			if (*beta) == zero {
				for i = 1; i <= leny; i++ {
					y.Set(i-1, zero)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(i-1, (*beta)*y.Get(i-1))
				}
			}
		} else {
			iy = ky
			if (*beta) == zero {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, zero)
					iy += (*incy)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy += (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	if trans == mat.NoTrans {
		//        Form  y := alpha*A*x + y.
		jx = kx
		if (*incy) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				for i = 1; i <= (*m); i++ {
					y.Set(i-1, y.Get(i-1)+temp*a.Get(i-1, j-1))
				}
				jx += (*incx)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				iy = ky
				for i = 1; i <= (*m); i++ {
					y.Set(iy-1, y.Get(iy-1)+temp*a.Get(i-1, j-1))
					iy += (*incy)
				}
				jx += (*incx)
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y.
		jy = ky
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = zero
				for i = 1; i <= (*m); i++ {
					temp += a.Get(i-1, j-1) * x.Get(i-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy += (*incy)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = zero
				ix = kx
				for i = 1; i <= (*m); i++ {
					temp += a.Get(i-1, j-1) * x.Get(ix-1)
					ix += (*incx)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy += (*incy)
			}
		}
	}
}

// Dsbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric band matrix, with k super-diagonals.
func Dsbmv(uplo mat.MatUplo, n, k *int, alpha *float64, a *mat.Matrix, lda *int, x *mat.Vector, incx *int, beta *float64, y *mat.Vector, incy *int) {
	var one, temp1, temp2, zero float64
	var i, info, ix, iy, j, jx, jy, kplus1, kx, ky, l int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*k) < 0 {
		info = 3
	} else if (*lda) < ((*k) + 1) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	} else if (*incy) == 0 {
		info = 11
	}
	if info != 0 {
		Xerbla([]byte("Dsbmv"), info)
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

	//     Start the operations. In this version the elements of the array A
	//     are accessed sequentially with one pass through A.
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
					iy += (*incy)
				}
			} else {
				for i = 1; i <= (*n); i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy += (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	if uplo == mat.Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = (*k) + 1
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				l = kplus1 - j
				for i = maxint(1, j-(*k)); i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(l+i-1, j-1))
					temp2 += a.Get(l+i-1, j-1) * x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(kplus1-1, j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				ix = kx
				iy = ky
				l = kplus1 - j
				for i = maxint(1, j-(*k)); i <= j-1; i++ {
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
					temp2 += a.Get(l+i-1, j-1) * x.Get(ix-1)
					ix += (*incx)
					iy += (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(kplus1-1, j-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
				if j > (*k) {
					kx += (*incx)
					ky += (*incy)
				}
			}
		}
	} else {
		//        Form  y  when lower triangle of A is stored.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*a.Get(0, j-1))
				l = 1 - j
				for i = j + 1; i <= minint((*n), j+(*k)); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(l+i-1, j-1))
					temp2 += a.Get(l+i-1, j-1) * x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(0, j-1))
				l = 1 - j
				ix = jx
				iy = jy
				for i = j + 1; i <= minint((*n), j+(*k)); i++ {
					ix += (*incx)
					iy += (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
					temp2 += a.Get(l+i-1, j-1) * x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
			}
		}
	}
}

// Dspmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix, supplied in packed form.
func Dspmv(uplo mat.MatUplo, n *int, alpha *float64, ap, x *mat.Vector, incx *int, beta *float64, y *mat.Vector, incy *int) {
	var one, temp1, temp2, zero float64
	var i, info, ix, iy, j, jx, jy, k, kk, kx, ky int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 6
	} else if (*incy) == 0 {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("Dspmv"), info)
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

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
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
					iy += (*incy)
				}
			} else {
				for i = 1; i <= (*n); i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy += (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	kk = 1
	if uplo == mat.Upper {
		//        Form  y  when AP contains the upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				k = kk
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*ap.Get(k-1))
					temp2 += ap.Get(k-1) * x.Get(i-1)
					k++
				}
				y.Set(j-1, y.Get(j-1)+temp1*ap.Get(kk+j-1-1)+(*alpha)*temp2)
				kk += j
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				ix = kx
				iy = ky
				for k = kk; k <= kk+j-2; k++ {
					y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
					temp2 += ap.Get(k-1) * x.Get(ix-1)
					ix += (*incx)
					iy += (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*ap.Get(kk+j-1-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
				kk += j
			}
		}
	} else {
		//
		//        Form  y  when AP contains the lower triangle.
		//
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*ap.Get(kk-1))
				k = kk + 1
				for i = j + 1; i <= (*n); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*ap.Get(k-1))
					temp2 += ap.Get(k-1) * x.Get(i-1)
					k++
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
				kk += (*n) - j + 1
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*ap.Get(kk-1))
				ix = jx
				iy = jy
				for k = kk + 1; k <= kk+(*n)-j; k++ {
					ix += (*incx)
					iy += (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
					temp2 += ap.Get(k-1) * x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
				kk += (*n) - j + 1
			}
		}
	}
}

// Dsymv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func Dsymv(uplo mat.MatUplo, n *int, alpha *float64, a *mat.Matrix, lda *int, x *mat.Vector, incx *int, beta *float64, y *mat.Vector, incy *int) {
	var one, temp1, temp2, zero float64
	var i, info, ix, iy, j, jx, jy, kx, ky int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*lda) < maxint(1, (*n)) {
		info = 5
	} else if (*incx) == 0 {
		info = 7
	} else if (*incy) == 0 {
		info = 10
	}
	if info != 0 {
		Xerbla([]byte("Dsymv"), info)
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
					iy += (*incy)
				}
			} else {
				for i = 1; i <= (*n); i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy += (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	if uplo == mat.Upper {
		//        Form  y  when A is stored in upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 += a.Get(i-1, j-1) * x.Get(i-1)
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
					temp2 += a.Get(i-1, j-1) * x.Get(ix-1)
					ix += (*incx)
					iy += (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
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
					temp2 += a.Get(i-1, j-1) * x.Get(i-1)
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
					ix += (*incx)
					iy += (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 += a.Get(i-1, j-1) * x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx += (*incx)
				jy += (*incy)
			}
		}
	}
}

// Dtbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func Dtbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k *int, a *mat.Matrix, lda *int, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, kplus1, kx, l int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*k) < 0 {
		info = 5
	} else if (*lda) < ((*k) + 1) {
		info = 7
	} else if (*incx) == 0 {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("Dtbmv"), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == mat.NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX   too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == mat.NoTrans {
		//         Form  x := A*x.
		if uplo == mat.Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						l = kplus1 - j
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							x.Set(i-1, x.Get(i-1)+temp*a.Get(l+i-1, j-1))
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*a.Get(kplus1-1, j-1))
						}
					}
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						l = kplus1 - j
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(l+i-1, j-1))
							ix += (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(kplus1-1, j-1))
						}
					}
					jx += (*incx)
					if j > (*k) {
						kx += (*incx)
					}
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						l = 1 - j
						for i = minint((*n), j+(*k)); i >= j+1; i-- {
							x.Set(i-1, x.Get(i-1)+temp*a.Get(l+i-1, j-1))
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*a.Get(0, j-1))
						}
					}
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						l = 1 - j
						for i = minint((*n), j+(*k)); i >= j+1; i-- {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(l+i-1, j-1))
							ix -= (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(0, j-1))
						}
					}
					jx -= (*incx)
					if ((*n) - j) >= (*k) {
						kx -= (*incx)
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					l = kplus1 - j
					if nounit {
						temp *= a.Get(kplus1-1, j-1)
					}
					for i = j - 1; i >= maxint(1, j-(*k)); i-- {
						temp += a.Get(l+i-1, j-1) * x.Get(i-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					kx -= (*incx)
					ix = kx
					l = kplus1 - j
					if nounit {
						temp *= a.Get(kplus1-1, j-1)
					}
					for i = j - 1; i >= maxint(1, j-(*k)); i-- {
						temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
						ix -= (*incx)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					l = 1 - j
					if nounit {
						temp *= a.Get(0, j-1)
					}
					for i = j + 1; i <= minint((*n), j+(*k)); i++ {
						temp += a.Get(l+i-1, j-1) * x.Get(i-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					kx += (*incx)
					ix = kx
					l = 1 - j
					if nounit {
						temp *= a.Get(0, j-1)
					}
					for i = j + 1; i <= minint((*n), j+(*k)); i++ {
						temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
						ix += (*incx)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
				}
			}
		}
	}
}

// Dtbsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular band matrix, with ( k + 1 )
// diagonals.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Dtbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k *int, a *mat.Matrix, lda *int, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, kplus1, kx, l int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*k) < 0 {
		info = 5
	} else if (*lda) < ((*k) + 1) {
		info = 7
	} else if (*incx) == 0 {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("Dtbsv"), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == mat.NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						l = kplus1 - j
						if nounit {
							x.Set(j-1, x.Get(j-1)/a.Get(kplus1-1, j-1))
						}
						temp = x.Get(j - 1)
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							x.Set(i-1, x.Get(i-1)-temp*a.Get(l+i-1, j-1))
						}
					}
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					kx -= (*incx)
					if x.Get(jx-1) != zero {
						ix = kx
						l = kplus1 - j
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(kplus1-1, j-1))
						}
						temp = x.Get(jx - 1)
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
							ix -= (*incx)
						}
					}
					jx -= (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						l = 1 - j
						if nounit {
							x.Set(j-1, x.Get(j-1)/a.Get(0, j-1))
						}
						temp = x.Get(j - 1)
						for i = j + 1; i <= minint((*n), j+(*k)); i++ {
							x.Set(i-1, x.Get(i-1)-temp*a.Get(l+i-1, j-1))
						}
					}
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					kx += (*incx)
					if x.Get(jx-1) != zero {
						ix = kx
						l = 1 - j
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(0, j-1))
						}
						temp = x.Get(jx - 1)
						for i = j + 1; i <= minint((*n), j+(*k)); i++ {
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
							ix += (*incx)
						}
					}
					jx += (*incx)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T)*x.
		if uplo == mat.Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					l = kplus1 - j
					for i = maxint(1, j-(*k)); i <= j-1; i++ {
						temp -= a.Get(l+i-1, j-1) * x.Get(i-1)
					}
					if nounit {
						temp /= a.Get(kplus1-1, j-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = kx
					l = kplus1 - j
					for i = maxint(1, j-(*k)); i <= j-1; i++ {
						temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
						ix += (*incx)
					}
					if nounit {
						temp /= a.Get(kplus1-1, j-1)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
					if j > (*k) {
						kx += (*incx)
					}
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					l = 1 - j
					for i = minint((*n), j+(*k)); i >= j+1; i-- {
						temp -= a.Get(l+i-1, j-1) * x.Get(i-1)
					}
					if nounit {
						temp /= a.Get(0, j-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = kx
					l = 1 - j
					for i = minint((*n), j+(*k)); i >= j+1; i-- {
						temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
						ix -= (*incx)
					}
					if nounit {
						temp /= a.Get(0, j-1)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
					if ((*n) - j) >= (*k) {
						kx -= (*incx)
					}
				}
			}
		}
	}
}

// Dtpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func Dtpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, ap, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, k, kk, kx int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*incx) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("Dtpmv"), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == mat.NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == mat.NoTrans {
		//        Form  x:= A*x.
		if uplo == mat.Upper {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						k = kk
						for i = 1; i <= j-1; i++ {
							x.Set(i-1, x.Get(i-1)+temp*ap.Get(k-1))
							k++
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*ap.Get(kk+j-1-1))
						}
					}
					kk += j
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for k = kk; k <= kk+j-2; k++ {
							x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
							ix += (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*ap.Get(kk+j-1-1))
						}
					}
					jx += (*incx)
					kk += j
				}
			}
		} else {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						k = kk
						for i = (*n); i >= j+1; i-- {
							x.Set(i-1, x.Get(i-1)+temp*ap.Get(k-1))
							k--
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*ap.Get(kk-(*n)+j-1))
						}
					}
					kk -= ((*n) - j + 1)
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for k = kk; k >= kk-((*n)-(j+1)); k-- {
							x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
							ix -= (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*ap.Get(kk-(*n)+j-1))
						}
					}
					jx -= (*incx)
					kk -= ((*n) - j + 1)
				}
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					k = kk - 1
					for i = j - 1; i >= 1; i-- {
						temp += ap.Get(k-1) * x.Get(i-1)
						k--
					}
					x.Set(j-1, temp)
					kk -= j
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = jx
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					for k = kk - 1; k >= kk-j+1; k-- {
						ix -= (*incx)
						temp += ap.Get(k-1) * x.Get(ix-1)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
					kk -= j
				}
			}
		} else {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					k = kk + 1
					for i = j + 1; i <= (*n); i++ {
						temp += ap.Get(k-1) * x.Get(i-1)
						k++
					}
					x.Set(j-1, temp)
					kk += ((*n) - j + 1)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = jx
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					for k = kk + 1; k <= kk+(*n)-j; k++ {
						ix += (*incx)
						temp += ap.Get(k-1) * x.Get(ix-1)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
					kk += ((*n) - j + 1)
				}
			}
		}
	}
}

// Dtpsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix, supplied in packed form.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Dtpsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, ap, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, k, kk, kx int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*incx) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("Dtpsv"), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == mat.NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						if nounit {
							x.Set(j-1, x.Get(j-1)/ap.Get(kk-1))
						}
						temp = x.Get(j - 1)
						k = kk - 1
						for i = j - 1; i >= 1; i-- {
							x.Set(i-1, x.Get(i-1)-temp*ap.Get(k-1))
							k--
						}
					}
					kk -= j
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/ap.Get(kk-1))
						}
						temp = x.Get(jx - 1)
						ix = jx
						for k = kk - 1; k >= kk-j+1; k-- {
							ix -= (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
						}
					}
					jx -= (*incx)
					kk -= j
				}
			}
		} else {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						if nounit {
							x.Set(j-1, x.Get(j-1)/ap.Get(kk-1))
						}
						temp = x.Get(j - 1)
						k = kk + 1
						for i = j + 1; i <= (*n); i++ {
							x.Set(i-1, x.Get(i-1)-temp*ap.Get(k-1))
							k++
						}
					}
					kk += ((*n) - j + 1)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/ap.Get(kk-1))
						}
						temp = x.Get(jx - 1)
						ix = jx
						for k = kk + 1; k <= kk+(*n)-j; k++ {
							ix += (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
						}
					}
					jx += (*incx)
					kk += ((*n) - j + 1)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x.
		if uplo == mat.Upper {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					k = kk
					for i = 1; i <= j-1; i++ {
						temp -= ap.Get(k-1) * x.Get(i-1)
						k++
					}
					if nounit {
						temp /= ap.Get(kk + j - 1 - 1)
					}
					x.Set(j-1, temp)
					kk += j
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = kx
					for k = kk; k <= kk+j-2; k++ {
						temp -= ap.Get(k-1) * x.Get(ix-1)
						ix += (*incx)
					}
					if nounit {
						temp /= ap.Get(kk + j - 1 - 1)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
					kk += j
				}
			}
		} else {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					k = kk
					for i = (*n); i >= j+1; i-- {
						temp -= ap.Get(k-1) * x.Get(i-1)
						k--
					}
					if nounit {
						temp /= ap.Get(kk - (*n) + j - 1)
					}
					x.Set(j-1, temp)
					kk -= ((*n) - j + 1)
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = kx
					for k = kk; k >= kk-((*n)-(j+1)); k-- {
						temp -= ap.Get(k-1) * x.Get(ix-1)
						ix -= (*incx)
					}
					if nounit {
						temp /= ap.Get(kk - (*n) + j - 1)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
					kk -= ((*n) - j + 1)
				}
			}
		}
	}
}

// Dtrmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func Dtrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, a *mat.Matrix, lda *int, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, kx int

	zero = 0.0
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*lda) < maxint(1, (*n)) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	}
	if info != 0 {
		Xerbla([]byte("Dtrmv"), info)
		return
	}
	//
	//     Quick return if possible.
	//
	if (*n) == 0 {
		return
	}
	//
	nounit = diag == mat.NonUnit
	//
	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	//
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}
	//
	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	if trans == mat.NoTrans {
		//
		//        Form  x := A*x.
		//
		if uplo == mat.Upper {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						for i = 1; i <= j-1; i++ {
							x.Set(i-1, x.Get(i-1)+temp*a.Get(i-1, j-1))
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*a.Get(j-1, j-1))
						}
					}
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for i = 1; i <= j-1; i++ {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(i-1, j-1))
							ix += (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
						}
					}
					jx += (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						for i = (*n); i >= j+1; i-- {
							x.Set(i-1, x.Get(i-1)+temp*a.Get(i-1, j-1))
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*a.Get(j-1, j-1))
						}
					}
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for i = (*n); i >= j+1; i-- {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(i-1, j-1))
							ix -= (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
						}
					}
					jx -= (*incx)
				}
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j - 1; i >= 1; i-- {
						temp += a.Get(i-1, j-1) * x.Get(i-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = jx
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j - 1; i >= 1; i-- {
						ix -= (*incx)
						temp += a.Get(i-1, j-1) * x.Get(ix-1)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j + 1; i <= (*n); i++ {
						temp += a.Get(i-1, j-1) * x.Get(i-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = jx
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j + 1; i <= (*n); i++ {
						ix += (*incx)
						temp += a.Get(i-1, j-1) * x.Get(ix-1)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
				}
			}
		}
	}
}

// Dtrsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Dtrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, a *mat.Matrix, lda *int, x *mat.Vector, incx *int) {
	var nounit bool
	var temp, zero float64
	var i, info, ix, j, jx, kx int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if !trans.IsValid() {
		info = 2
	} else if !diag.IsValid() {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*lda) < maxint(1, (*n)) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	}
	if info != 0 {
		Xerbla([]byte("Dtrsv"), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == mat.NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						if nounit {
							x.Set(j-1, x.Get(j-1)/a.Get(j-1, j-1))
						}
						temp = x.Get(j - 1)
						for i = j - 1; i >= 1; i-- {
							x.Set(i-1, x.Get(i-1)-temp*a.Get(i-1, j-1))
						}
					}
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(j-1, j-1))
						}
						temp = x.Get(jx - 1)
						ix = jx
						for i = j - 1; i >= 1; i-- {
							ix -= (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
						}
					}
					jx -= (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						if nounit {
							x.Set(j-1, x.Get(j-1)/a.Get(j-1, j-1))
						}
						temp = x.Get(j - 1)
						for i = j + 1; i <= (*n); i++ {
							x.Set(i-1, x.Get(i-1)-temp*a.Get(i-1, j-1))
						}
					}
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(j-1, j-1))
						}
						temp = x.Get(jx - 1)
						ix = jx
						for i = j + 1; i <= (*n); i++ {
							ix += (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
						}
					}
					jx += (*incx)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x.
		if uplo == mat.Upper {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					for i = 1; i <= j-1; i++ {
						temp -= a.Get(i-1, j-1) * x.Get(i-1)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = kx
					for i = 1; i <= j-1; i++ {
						temp -= a.Get(i-1, j-1) * x.Get(ix-1)
						ix += (*incx)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
					x.Set(jx-1, temp)
					jx += (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					for i = (*n); i >= j+1; i-- {
						temp -= a.Get(i-1, j-1) * x.Get(i-1)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
					x.Set(j-1, temp)
				}
			} else {
				kx += ((*n) - 1) * (*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = kx
					for i = (*n); i >= j+1; i-- {
						temp -= a.Get(i-1, j-1) * x.Get(ix-1)
						ix -= (*incx)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
					x.Set(jx-1, temp)
					jx -= (*incx)
				}
			}
		}
	}
}

// Dger performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Dger(m, n *int, alpha *float64, x *mat.Vector, incx *int, y *mat.Vector, incy *int, a *mat.Matrix, lda *int) {
	var temp, zero float64
	var i, info, ix, j, jy, kx int

	zero = 0.0
	//     Test the input parameters.
	info = 0
	if (*m) < 0 {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*incy) == 0 {
		info = 7
	} else if (*lda) < maxint(1, (*m)) {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("Dger"), info)
		return
	}

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) || ((*alpha) == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if (*incy) > 0 {
		jy = 1
	} else {
		jy = 1 - ((*n)-1)*(*incy)
	}
	if (*incx) == 1 {
		for j = 1; j <= (*n); j++ {
			if y.Get(jy-1) != zero {
				temp = (*alpha) * y.Get(jy-1)
				for i = 1; i <= (*m); i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
				}
			}
			jy += (*incy)
		}
	} else {
		if (*incx) > 0 {
			kx = 1
		} else {
			kx = 1 - ((*m)-1)*(*incx)
		}
		for j = 1; j <= (*n); j++ {
			if y.Get(jy-1) != zero {
				temp = (*alpha) * y.Get(jy-1)
				ix = kx
				for i = 1; i <= (*m); i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
					ix += (*incx)
				}
			}
			jy += (*incy)
		}
	}
}

// Dspr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr(uplo mat.MatUplo, n *int, alpha *float64, x *mat.Vector, incx *int, ap *mat.Vector) {
	var temp, zero float64
	var i, info, ix, j, jx, k, kk, kx int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	}
	if info != 0 {
		Xerbla([]byte("Dspr"), info)
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

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == mat.Upper {
		//        Form  A  when upper triangle is stored in AP.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = (*alpha) * x.Get(j-1)
					k = kk
					for i = 1; i <= j; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k++
					}
				}
				kk += j
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = (*alpha) * x.Get(jx-1)
					ix = kx
					for k = kk; k <= kk+j-1; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
						ix += (*incx)
					}
				}
				jx += (*incx)
				kk += j
			}
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = (*alpha) * x.Get(j-1)
					k = kk
					for i = j; i <= (*n); i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k++
					}
				}
				kk += (*n) - j + 1
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = (*alpha) * x.Get(jx-1)
					ix = jx
					for k = kk; k <= kk+(*n)-j; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
						ix += (*incx)
					}
				}
				jx += (*incx)
				kk += (*n) - j + 1
			}
		}
	}
}

// Dsyr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func Dsyr(uplo mat.MatUplo, n *int, alpha *float64, x *mat.Vector, incx *int, a *mat.Matrix, lda *int) {
	var temp, zero float64
	var i, info, ix, j, jx, kx int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*lda) < maxint(1, (*n)) {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("Dsyr"), info)
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
	if uplo == mat.Upper {
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
						ix += (*incx)
					}
				}
				jx += (*incx)
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
						ix += (*incx)
					}
				}
				jx += (*incx)
			}
		}
	}
}

// Dspr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr2(uplo mat.MatUplo, n *int, alpha *float64, x *mat.Vector, incx *int, y *mat.Vector, incy *int, ap *mat.Vector) {
	var temp1, temp2, zero float64
	var i, info, ix, iy, j, jx, jy, k, kk, kx, ky int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*incy) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("Dspr2"), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || ((*alpha) == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	if ((*incx) != 1) || ((*incy) != 1) {
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
		jx = kx
		jy = ky
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == mat.Upper {
		//        Form  A  when upper triangle is stored in AP.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.Get(j-1)
					temp2 = (*alpha) * x.Get(j-1)
					k = kk
					for i = 1; i <= j; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
						k++
					}
				}
				kk += j
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.Get(jy-1)
					temp2 = (*alpha) * x.Get(jx-1)
					ix = kx
					iy = ky
					for k = kk; k <= kk+j-1; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix += (*incx)
						iy += (*incy)
					}
				}
				jx += (*incx)
				jy += (*incy)
				kk += j
			}
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.Get(j-1)
					temp2 = (*alpha) * x.Get(j-1)
					k = kk
					for i = j; i <= (*n); i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
						k++
					}
				}
				kk += (*n) - j + 1
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.Get(jy-1)
					temp2 = (*alpha) * x.Get(jx-1)
					ix = jx
					iy = jy
					for k = kk; k <= kk+(*n)-j; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix += (*incx)
						iy += (*incy)
					}
				}
				jx += (*incx)
				jy += (*incy)
				kk += (*n) - j + 1
			}
		}
	}
}

// Dsyr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n symmetric matrix.
func Dsyr2(uplo mat.MatUplo, n *int, alpha *float64, x *mat.Vector, incx *int, y *mat.Vector, incy *int, a *mat.Matrix, lda *int) {
	var temp1, temp2, zero float64
	var i, info, ix, iy, j, jx, jy, kx, ky int

	zero = 0.0

	//     Test the input parameters.
	info = 0
	if !uplo.IsValid() {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*incy) == 0 {
		info = 7
	} else if (*lda) < maxint(1, (*n)) {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("Dsyr2"), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || ((*alpha) == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	if ((*incx) != 1) || ((*incy) != 1) {
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
		jx = kx
		jy = ky
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == mat.Upper {
		//        Form  A  when A is stored in the upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.Get(j-1)
					temp2 = (*alpha) * x.Get(j-1)
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.Get(jy-1)
					temp2 = (*alpha) * x.Get(jx-1)
					ix = kx
					iy = ky
					for i = 1; i <= j; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix += (*incx)
						iy += (*incy)
					}
				}
				jx += (*incx)
				jy += (*incy)
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.Get(j-1)
					temp2 = (*alpha) * x.Get(j-1)
					for i = j; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.Get(jy-1)
					temp2 = (*alpha) * x.Get(jx-1)
					ix = jx
					iy = jy
					for i = j; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix += (*incx)
						iy += (*incy)
					}
				}
				jx += (*incx)
				jy += (*incy)
			}
		}
	}
}
