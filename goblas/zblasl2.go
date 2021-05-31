package goblas

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zgbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//
//    y := alpha*A**H*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func Zgbmv(trans mat.MatTrans, m, n, kl, ku *int, alpha *complex128, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var noconj bool
	var one, temp, zero complex128
	var i, info, ix, iy, j, jx, jy, k, kup1, kx, ky, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if trans != NoTrans && trans != Trans && trans != ConjTrans {
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
		Xerbla([]byte("ZGBMV "), info)
		return
	}

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) || (((*alpha) == zero) && ((*beta) == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
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
					iy = iy + (*incy)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy = iy + (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	kup1 = (*ku) + 1
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		jx = kx
		if (*incy) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
					y.Set(i-1, y.Get(i-1)+temp*a.Get(k+i-1, j-1))
				}
				jx = jx + (*incx)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				iy = ky
				k = kup1 - j
				for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
					y.Set(iy-1, y.Get(iy-1)+temp*a.Get(k+i-1, j-1))
					iy = iy + (*incy)
				}
				jx = jx + (*incx)
				if j > (*ku) {
					ky = ky + (*incy)
				}
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		jy = ky
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = zero
				k = kup1 - j
				if noconj {
					for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
						temp = temp + a.Get(k+i-1, j-1)*x.Get(i-1)
					}
				} else {
					for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
						temp = temp + a.GetConj(k+i-1, j-1)*x.Get(i-1)
					}
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy = jy + (*incy)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = zero
				ix = kx
				k = kup1 - j
				if noconj {
					for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
						temp = temp + a.Get(k+i-1, j-1)*x.Get(ix-1)
						ix = ix + (*incx)
					}
				} else {
					for i = maxint(1, j-(*ku)); i <= minint(*m, j+(*kl)); i++ {
						temp = temp + a.GetConj(k+i-1, j-1)*x.Get(ix-1)
						ix = ix + (*incx)
					}
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy = jy + (*incy)
				if j > (*ku) {
					kx = kx + (*incx)
				}
			}
		}
	}
}

// Zgemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//
//    y := alpha*A**H*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func Zgemv(trans mat.MatTrans, m, n *int, alpha *complex128, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var noconj bool
	var one, temp, zero complex128
	var i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 1
	} else if (*m) < 0 {
		info = 2
	} else if (*n) < 0 {
		info = 3
	} else if (*lda) < maxint(1, *m) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	} else if (*incy) == 0 {
		info = 11
	}
	if info != 0 {
		Xerbla([]byte("ZGEMV "), info)
		return
	}

	//     Quick return if possible.
	if ((*m) == 0) || ((*n) == 0) || (((*alpha) == zero) && ((*beta) == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
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
					iy = iy + (*incy)
				}
			} else {
				for i = 1; i <= leny; i++ {
					y.Set(iy-1, (*beta)*y.Get(iy-1))
					iy = iy + (*incy)
				}
			}
		}
	}
	if (*alpha) == zero {
		return
	}
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		jx = kx
		if (*incy) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				for i = 1; i <= (*m); i++ {
					y.Set(i-1, y.Get(i-1)+temp*a.Get(i-1, j-1))
				}
				jx = jx + (*incx)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = (*alpha) * x.Get(jx-1)
				iy = ky
				for i = 1; i <= (*m); i++ {
					y.Set(iy-1, y.Get(iy-1)+temp*a.Get(i-1, j-1))
					iy = iy + (*incy)
				}
				jx = jx + (*incx)
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		jy = ky
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				temp = zero
				if noconj {
					for i = 1; i <= (*m); i++ {
						temp = temp + a.Get(i-1, j-1)*x.Get(i-1)
					}
				} else {
					for i = 1; i <= (*m); i++ {
						temp = temp + a.GetConj(i-1, j-1)*x.Get(i-1)
					}
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy = jy + (*incy)
			}
		} else {
			for j = 1; j <= (*n); j++ {
				temp = zero
				ix = kx
				if noconj {
					for i = 1; i <= (*m); i++ {
						temp = temp + a.Get(i-1, j-1)*x.Get(ix-1)
						ix = ix + (*incx)
					}
				} else {
					for i = 1; i <= (*m); i++ {
						temp = temp + a.GetConj(i-1, j-1)*x.Get(ix-1)
						ix = ix + (*incx)
					}
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp)
				jy = jy + (*incy)
			}
		}
	}
}

// Zhbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian band matrix, with k super-diagonals.
func Zhbmv(uplo mat.MatUplo, n, k *int, alpha *complex128, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var one, temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, kplus1, kx, ky, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
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
		Xerbla([]byte("ZHBMV "), info)
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
	if uplo == Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = (*k) + 1
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				l = kplus1 - j
				for i = maxint(1, j-(*k)); i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(l+i-1, j-1))
					temp2 = temp2 + a.GetConj(l+i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+temp1*a.GetReCmplx(kplus1-1, j-1)+(*alpha)*temp2)
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
					temp2 = temp2 + a.GetConj(l+i-1, j-1)*x.Get(ix-1)
					ix = ix + (*incx)
					iy = iy + (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(kplus1-1, j-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
				if j > (*k) {
					kx = kx + (*incx)
					ky = ky + (*incy)
				}
			}
		}
	} else {
		//        Form  y  when lower triangle of A is stored.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*a.GetReCmplx(0, j-1))
				l = 1 - j
				for i = j + 1; i <= minint(*n, j+(*k)); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(l+i-1, j-1))
					temp2 = temp2 + a.GetConj(l+i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(0, j-1))
				l = 1 - j
				ix = jx
				iy = jy
				for i = j + 1; i <= minint(*n, j+(*k)); i++ {
					ix = ix + (*incx)
					iy = iy + (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
					temp2 = temp2 + a.GetConj(l+i-1, j-1)*x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	}
}

// Zhemv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix.
func Zhemv(uplo mat.MatUplo, n *int, alpha *complex128, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var one, temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
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
		Xerbla([]byte("ZHEMV "), info)
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
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.GetConj(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+temp1*a.GetReCmplx(j-1, j-1)+(*alpha)*temp2)
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
					temp2 = temp2 + a.GetConj(i-1, j-1)*x.Get(ix-1)
					ix = ix + (*incx)
					iy = iy + (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(j-1, j-1)+(*alpha)*temp2)
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
				y.Set(j-1, y.Get(j-1)+temp1*a.GetReCmplx(j-1, j-1))
				for i = j + 1; i <= (*n); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.GetConj(i-1, j-1)*x.Get(i-1)
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(j-1, j-1))
				ix = jx
				iy = jy
				for i = j + 1; i <= (*n); i++ {
					ix = ix + (*incx)
					iy = iy + (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
					temp2 = temp2 + a.GetConj(i-1, j-1)*x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	}
}

// Zhpmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix, supplied in packed form.
func Zhpmv(uplo mat.MatUplo, n *int, alpha *complex128, ap, x *mat.CVector, incx *int, beta *complex128, y *mat.CVector, incy *int) {
	var one, temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, k, kk, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 6
	} else if (*incy) == 0 {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("ZHPMV "), info)
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
	kk = 1
	if uplo == Upper {
		//        Form  y  when AP contains the upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				k = kk
				for i = 1; i <= j-1; i++ {
					y.Set(i-1, y.Get(i-1)+temp1*ap.Get(k-1))
					temp2 = temp2 + ap.GetConj(k-1)*x.Get(i-1)
					k = k + 1
				}
				y.Set(j-1, y.Get(j-1)+temp1*ap.GetReCmplx(kk+j-1-1)+(*alpha)*temp2)
				kk = kk + j
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
					temp2 = temp2 + ap.GetConj(k-1)*x.Get(ix-1)
					ix = ix + (*incx)
					iy = iy + (*incy)
				}
				y.Set(jy-1, y.Get(jy-1)+temp1*ap.GetReCmplx(kk+j-1-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
				kk = kk + j
			}
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(j-1)
				temp2 = zero
				y.Set(j-1, y.Get(j-1)+temp1*ap.GetReCmplx(kk-1))
				k = kk + 1
				for i = j + 1; i <= (*n); i++ {
					y.Set(i-1, y.Get(i-1)+temp1*ap.Get(k-1))
					temp2 = temp2 + ap.GetConj(k-1)*x.Get(i-1)
					k = k + 1
				}
				y.Set(j-1, y.Get(j-1)+(*alpha)*temp2)
				kk = kk + ((*n) - j + 1)
			}
		} else {
			jx = kx
			jy = ky
			for j = 1; j <= (*n); j++ {
				temp1 = (*alpha) * x.Get(jx-1)
				temp2 = zero
				y.Set(jy-1, y.Get(jy-1)+temp1*ap.GetReCmplx(kk-1))
				ix = jx
				iy = jy
				for k = kk + 1; k <= kk+(*n)-j; k++ {
					ix = ix + (*incx)
					iy = iy + (*incy)
					y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
					temp2 = temp2 + ap.GetConj(k-1)*x.Get(ix-1)
				}
				y.Set(jy-1, y.Get(jy-1)+(*alpha)*temp2)
				jx = jx + (*incx)
				jy = jy + (*incy)
				kk = kk + ((*n) - j + 1)
			}
		}
	}
}

// Ztbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func Ztbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k *int, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, kplus1, kx, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
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
		Xerbla([]byte("ZTBMV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX   too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//         Form  x := A*x.
		if uplo == Upper {
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
							ix = ix + (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(kplus1-1, j-1))
						}
					}
					jx = jx + (*incx)
					if j > (*k) {
						kx = kx + (*incx)
					}
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						l = 1 - j
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							x.Set(i-1, x.Get(i-1)+temp*a.Get(l+i-1, j-1))
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*a.Get(0, j-1))
						}
					}
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						l = 1 - j
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(l+i-1, j-1))
							ix = ix - (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(0, j-1))
						}
					}
					jx = jx - (*incx)
					if ((*n) - j) >= (*k) {
						kx = kx - (*incx)
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					l = kplus1 - j
					if noconj {
						if nounit {
							temp = temp * a.Get(kplus1-1, j-1)
						}
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							temp = temp + a.Get(l+i-1, j-1)*x.Get(i-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(kplus1-1, j-1)
						}
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							temp = temp + a.GetConj(l+i-1, j-1)*x.Get(i-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					kx = kx - (*incx)
					ix = kx
					l = kplus1 - j
					if noconj {
						if nounit {
							temp = temp * a.Get(kplus1-1, j-1)
						}
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							temp = temp + a.Get(l+i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(kplus1-1, j-1)
						}
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							temp = temp + a.GetConj(l+i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					l = 1 - j
					if noconj {
						if nounit {
							temp = temp * a.Get(0, j-1)
						}
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							temp = temp + a.Get(l+i-1, j-1)*x.Get(i-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(0, j-1)
						}
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							temp = temp + a.GetConj(l+i-1, j-1)*x.Get(i-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					kx = kx + (*incx)
					ix = kx
					l = 1 - j
					if noconj {
						if nounit {
							temp = temp * a.Get(0, j-1)
						}
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							temp = temp + a.Get(l+i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(0, j-1)
						}
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							temp = temp + a.GetConj(l+i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
				}
			}
		}
	}
}

// Ztbsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular band matrix, with ( k + 1 )
// diagonals.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Ztbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k *int, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, kplus1, kx, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
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
		Xerbla([]byte("ZTBSV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
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
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					kx = kx - (*incx)
					if x.Get(jx-1) != zero {
						ix = kx
						l = kplus1 - j
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(kplus1-1, j-1))
						}
						temp = x.Get(jx - 1)
						for i = j - 1; i >= maxint(1, j-(*k)); i-- {
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
							ix = ix - (*incx)
						}
					}
					jx = jx - (*incx)
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
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							x.Set(i-1, x.Get(i-1)-temp*a.Get(l+i-1, j-1))
						}
					}
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					kx = kx + (*incx)
					if x.Get(jx-1) != zero {
						ix = kx
						l = 1 - j
						if nounit {
							x.Set(jx-1, x.Get(jx-1)/a.Get(0, j-1))
						}
						temp = x.Get(jx - 1)
						for i = j + 1; i <= minint(*n, j+(*k)); i++ {
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
							ix = ix + (*incx)
						}
					}
					jx = jx + (*incx)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kplus1 = (*k) + 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					l = kplus1 - j
					if noconj {
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							temp = temp - a.Get(l+i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.Get(kplus1-1, j-1)
						}
					} else {
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							temp = temp - a.GetConj(l+i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.GetConj(kplus1-1, j-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = kx
					l = kplus1 - j
					if noconj {
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							temp = temp - a.Get(l+i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / a.Get(kplus1-1, j-1)
						}
					} else {
						for i = maxint(1, j-(*k)); i <= j-1; i++ {
							temp = temp - a.GetConj(l+i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / a.GetConj(kplus1-1, j-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
					if j > (*k) {
						kx = kx + (*incx)
					}
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					l = 1 - j
					if noconj {
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							temp = temp - a.Get(l+i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.Get(0, j-1)
						}
					} else {
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							temp = temp - a.GetConj(l+i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.GetConj(0, j-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = kx
					l = 1 - j
					if noconj {
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							temp = temp - a.Get(l+i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / a.Get(0, j-1)
						}
					} else {
						for i = minint(*n, j+(*k)); i >= j+1; i-- {
							temp = temp - a.GetConj(l+i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / a.GetConj(0, j-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
					if ((*n) - j) >= (*k) {
						kx = kx - (*incx)
					}
				}
			}
		}
	}
}

// Ztpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func Ztpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, ap, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*incx) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("ZTPMV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x:= A*x.
		if uplo == Upper {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					if x.Get(j-1) != zero {
						temp = x.Get(j - 1)
						k = kk
						for i = 1; i <= j-1; i++ {
							x.Set(i-1, x.Get(i-1)+temp*ap.Get(k-1))
							k = k + 1
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*ap.Get(kk+j-1-1))
						}
					}
					kk = kk + j
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for k = kk; k <= kk+j-2; k++ {
							x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
							ix = ix + (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*ap.Get(kk+j-1-1))
						}
					}
					jx = jx + (*incx)
					kk = kk + j
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
							k = k - 1
						}
						if nounit {
							x.Set(j-1, x.Get(j-1)*ap.Get(kk-(*n)+j-1))
						}
					}
					kk = kk - ((*n) - j + 1)
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for k = kk; k >= kk-((*n)-(j+1)); k-- {
							x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
							ix = ix - (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*ap.Get(kk-(*n)+j-1))
						}
					}
					jx = jx - (*incx)
					kk = kk - ((*n) - j + 1)
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					k = kk - 1
					if noconj {
						if nounit {
							temp = temp * ap.Get(kk-1)
						}
						for i = j - 1; i >= 1; i-- {
							temp = temp + ap.Get(k-1)*x.Get(i-1)
							k = k - 1
						}
					} else {
						if nounit {
							temp = temp * ap.GetConj(kk-1)
						}
						for i = j - 1; i >= 1; i-- {
							temp = temp + ap.GetConj(k-1)*x.Get(i-1)
							k = k - 1
						}
					}
					x.Set(j-1, temp)
					kk = kk - j
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = jx
					if noconj {
						if nounit {
							temp = temp * ap.Get(kk-1)
						}
						for k = kk - 1; k >= kk-j+1; k-- {
							ix = ix - (*incx)
							temp = temp + ap.Get(k-1)*x.Get(ix-1)
						}
					} else {
						if nounit {
							temp = temp * ap.GetConj(kk-1)
						}
						for k = kk - 1; k >= kk-j+1; k-- {
							ix = ix - (*incx)
							temp = temp + ap.GetConj(k-1)*x.Get(ix-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
					kk = kk - j
				}
			}
		} else {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					k = kk + 1
					if noconj {
						if nounit {
							temp = temp * ap.Get(kk-1)
						}
						for i = j + 1; i <= (*n); i++ {
							temp = temp + ap.Get(k-1)*x.Get(i-1)
							k = k + 1
						}
					} else {
						if nounit {
							temp = temp * ap.GetConj(kk-1)
						}
						for i = j + 1; i <= (*n); i++ {
							temp = temp + ap.GetConj(k-1)*x.Get(i-1)
							k = k + 1
						}
					}
					x.Set(j-1, temp)
					kk = kk + ((*n) - j + 1)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = jx
					if noconj {
						if nounit {
							temp = temp * ap.Get(kk-1)
						}
						for k = kk + 1; k <= kk+(*n)-j; k++ {
							ix = ix + (*incx)
							temp = temp + ap.Get(k-1)*x.Get(ix-1)
						}
					} else {
						if nounit {
							temp = temp * ap.GetConj(kk-1)
						}
						for k = kk + 1; k <= kk+(*n)-j; k++ {
							ix = ix + (*incx)
							temp = temp + ap.GetConj(k-1)*x.Get(ix-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
					kk = kk + ((*n) - j + 1)
				}
			}
		}
	}
}

// Ztpsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix, supplied in packed form.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Ztpsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, ap, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*incx) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("ZTPSV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
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
							k = k - 1
						}
					}
					kk = kk - j
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
							ix = ix - (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
						}
					}
					jx = jx - (*incx)
					kk = kk - j
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
							k = k + 1
						}
					}
					kk = kk + ((*n) - j + 1)
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
							ix = ix + (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
						}
					}
					jx = jx + (*incx)
					kk = kk + ((*n) - j + 1)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kk = 1
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					k = kk
					if noconj {
						for i = 1; i <= j-1; i++ {
							temp = temp - ap.Get(k-1)*x.Get(i-1)
							k = k + 1
						}
						if nounit {
							temp = temp / ap.Get(kk+j-1-1)
						}
					} else {
						for i = 1; i <= j-1; i++ {
							temp = temp - ap.GetConj(k-1)*x.Get(i-1)
							k = k + 1
						}
						if nounit {
							temp = temp / ap.GetConj(kk+j-1-1)
						}
					}
					x.Set(j-1, temp)
					kk = kk + j
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = kx
					if noconj {
						for k = kk; k <= kk+j-2; k++ {
							temp = temp - ap.Get(k-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / ap.Get(kk+j-1-1)
						}
					} else {
						for k = kk; k <= kk+j-2; k++ {
							temp = temp - ap.GetConj(k-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / ap.GetConj(kk+j-1-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
					kk = kk + j
				}
			}
		} else {
			kk = ((*n) * ((*n) + 1)) / 2
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					k = kk
					if noconj {
						for i = (*n); i >= j+1; i-- {
							temp = temp - ap.Get(k-1)*x.Get(i-1)
							k = k - 1
						}
						if nounit {
							temp = temp / ap.Get(kk-(*n)+j-1)
						}
					} else {
						for i = (*n); i >= j+1; i-- {
							temp = temp - ap.GetConj(k-1)*x.Get(i-1)
							k = k - 1
						}
						if nounit {
							temp = temp / ap.GetConj(kk-(*n)+j-1)
						}
					}
					x.Set(j-1, temp)
					kk = kk - ((*n) - j + 1)
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = kx
					if noconj {
						for k = kk; k >= kk-((*n)-(j+1)); k-- {
							temp = temp - ap.Get(k-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / ap.Get(kk-(*n)+j-1)
						}
					} else {
						for k = kk; k >= kk-((*n)-(j+1)); k-- {
							temp = temp - ap.GetConj(k-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / ap.GetConj(kk-(*n)+j-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
					kk = kk - ((*n) - j + 1)
				}
			}
		}
	}
}

// Ztrmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func Ztrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*lda) < maxint(1, *n) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	}
	if info != 0 {
		Xerbla([]byte("ZTRMV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := A*x.
		if uplo == Upper {
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
							ix = ix + (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
						}
					}
					jx = jx + (*incx)
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
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					if x.Get(jx-1) != zero {
						temp = x.Get(jx - 1)
						ix = kx
						for i = (*n); i >= j+1; i-- {
							x.Set(ix-1, x.Get(ix-1)+temp*a.Get(i-1, j-1))
							ix = ix - (*incx)
						}
						if nounit {
							x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
						}
					}
					jx = jx - (*incx)
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					if noconj {
						if nounit {
							temp = temp * a.Get(j-1, j-1)
						}
						for i = j - 1; i >= 1; i-- {
							temp = temp + a.Get(i-1, j-1)*x.Get(i-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(j-1, j-1)
						}
						for i = j - 1; i >= 1; i-- {
							temp = temp + a.GetConj(i-1, j-1)*x.Get(i-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx + ((*n)-1)*(*incx)
				for j = (*n); j >= 1; j-- {
					temp = x.Get(jx - 1)
					ix = jx
					if noconj {
						if nounit {
							temp = temp * a.Get(j-1, j-1)
						}
						for i = j - 1; i >= 1; i-- {
							ix = ix - (*incx)
							temp = temp + a.Get(i-1, j-1)*x.Get(ix-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(j-1, j-1)
						}
						for i = j - 1; i >= 1; i-- {
							ix = ix - (*incx)
							temp = temp + a.GetConj(i-1, j-1)*x.Get(ix-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					if noconj {
						if nounit {
							temp = temp * a.Get(j-1, j-1)
						}
						for i = j + 1; i <= (*n); i++ {
							temp = temp + a.Get(i-1, j-1)*x.Get(i-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(j-1, j-1)
						}
						for i = j + 1; i <= (*n); i++ {
							temp = temp + a.GetConj(i-1, j-1)*x.Get(i-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					temp = x.Get(jx - 1)
					ix = jx
					if noconj {
						if nounit {
							temp = temp * a.Get(j-1, j-1)
						}
						for i = j + 1; i <= (*n); i++ {
							ix = ix + (*incx)
							temp = temp + a.Get(i-1, j-1)*x.Get(ix-1)
						}
					} else {
						if nounit {
							temp = temp * a.GetConj(j-1, j-1)
						}
						for i = j + 1; i <= (*n); i++ {
							ix = ix + (*incx)
							temp = temp + a.GetConj(i-1, j-1)*x.Get(ix-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
				}
			}
		}
	}
}

// Ztrsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Ztrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n *int, a *mat.CMatrix, lda *int, x *mat.CVector, incx *int) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, info, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		info = 2
	} else if diag != Unit && diag != NonUnit {
		info = 3
	} else if (*n) < 0 {
		info = 4
	} else if (*lda) < maxint(1, *n) {
		info = 6
	} else if (*incx) == 0 {
		info = 8
	}
	if info != 0 {
		Xerbla([]byte("ZTRSV "), info)
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if (*incx) <= 0 {
		kx = 1 - ((*n)-1)*(*incx)
	} else if (*incx) != 1 {
		kx = 1
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
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
							ix = ix - (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
						}
					}
					jx = jx - (*incx)
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
							ix = ix + (*incx)
							x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
						}
					}
					jx = jx + (*incx)
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			if (*incx) == 1 {
				for j = 1; j <= (*n); j++ {
					temp = x.Get(j - 1)
					if noconj {
						for i = 1; i <= j-1; i++ {
							temp = temp - a.Get(i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.Get(j-1, j-1)
						}
					} else {
						for i = 1; i <= j-1; i++ {
							temp = temp - a.GetConj(i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.GetConj(j-1, j-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				jx = kx
				for j = 1; j <= (*n); j++ {
					ix = kx
					temp = x.Get(jx - 1)
					if noconj {
						for i = 1; i <= j-1; i++ {
							temp = temp - a.Get(i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / a.Get(j-1, j-1)
						}
					} else {
						for i = 1; i <= j-1; i++ {
							temp = temp - a.GetConj(i-1, j-1)*x.Get(ix-1)
							ix = ix + (*incx)
						}
						if nounit {
							temp = temp / a.GetConj(j-1, j-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx + (*incx)
				}
			}
		} else {
			if (*incx) == 1 {
				for j = (*n); j >= 1; j-- {
					temp = x.Get(j - 1)
					if noconj {
						for i = (*n); i >= j+1; i-- {
							temp = temp - a.Get(i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.Get(j-1, j-1)
						}
					} else {
						for i = (*n); i >= j+1; i-- {
							temp = temp - a.GetConj(i-1, j-1)*x.Get(i-1)
						}
						if nounit {
							temp = temp / a.GetConj(j-1, j-1)
						}
					}
					x.Set(j-1, temp)
				}
			} else {
				kx = kx + ((*n)-1)*(*incx)
				jx = kx
				for j = (*n); j >= 1; j-- {
					ix = kx
					temp = x.Get(jx - 1)
					if noconj {
						for i = (*n); i >= j+1; i-- {
							temp = temp - a.Get(i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / a.Get(j-1, j-1)
						}
					} else {
						for i = (*n); i >= j+1; i-- {
							temp = temp - a.GetConj(i-1, j-1)*x.Get(ix-1)
							ix = ix - (*incx)
						}
						if nounit {
							temp = temp / a.GetConj(j-1, j-1)
						}
					}
					x.Set(jx-1, temp)
					jx = jx - (*incx)
				}
			}
		}
	}
}

// Zgerc performs the rank 1 operation
//
//    A := alpha*x*y**H + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Zgerc(m, n *int, alpha *complex128, x *mat.CVector, incx *int, y *mat.CVector, incy *int, a *mat.CMatrix, lda *int) {
	var temp, zero complex128
	var i, info, ix, j, jy, kx int

	zero = (0.0 + 0.0*1i)

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
	} else if (*lda) < maxint(1, *m) {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("ZGERC "), info)
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
				temp = (*alpha) * y.GetConj(jy-1)
				for i = 1; i <= (*m); i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
				}
			}
			jy = jy + (*incy)
		}
	} else {
		if (*incx) > 0 {
			kx = 1
		} else {
			kx = 1 - ((*m)-1)*(*incx)
		}
		for j = 1; j <= (*n); j++ {
			if y.Get(jy-1) != zero {
				temp = (*alpha) * y.GetConj(jy-1)
				ix = kx
				for i = 1; i <= (*m); i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
					ix = ix + (*incx)
				}
			}
			jy = jy + (*incy)
		}
	}
}

// Zgeru performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Zgeru(m, n *int, alpha *complex128, x *mat.CVector, incx *int, y *mat.CVector, incy *int, a *mat.CMatrix, lda *int) {
	var temp, zero complex128
	var i, info, ix, j, jy, kx int

	zero = (0.0 + 0.0*1i)

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
	} else if (*lda) < maxint(1, *m) {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("ZGERU "), info)
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
			jy = jy + (*incy)
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
					ix = ix + (*incx)
				}
			}
			jy = jy + (*incy)
		}
	}
}

// Zher performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix.
func Zher(uplo mat.MatUplo, n *int, alpha *float64, x *mat.CVector, incx *int, a *mat.CMatrix, lda *int) {
	var temp, zero complex128
	var i, info, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*lda) < maxint(1, *n) {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("ZHER  "), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || ((*alpha) == real(zero)) {
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
	if uplo == Upper {
		//        Form  A  when A is stored in upper triangle.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(j-1)
					for i = 1; i <= j-1; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(j-1)*temp), 0))
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(jx-1)
					ix = kx
					for i = 1; i <= j-1; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
						ix = ix + (*incx)
					}
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp), 0))
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
				jx = jx + (*incx)
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(j-1)
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(temp*x.Get(j-1)), 0))
					for i = j + 1; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp)
					}
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(jx-1)
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(temp*x.Get(jx-1)), 0))
					ix = jx
					for i = j + 1; i <= (*n); i++ {
						ix = ix + (*incx)
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
					}
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
				jx = jx + (*incx)
			}
		}
	}
}

// Zhpr performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix, supplied in packed form.
func Zhpr(uplo mat.MatUplo, n *int, alpha *float64, x *mat.CVector, incx *int, ap *mat.CVector) {
	var temp, zero complex128
	var i, info, ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	}
	if info != 0 {
		Xerbla([]byte("ZHPR  "), info)
		return
	}

	//     Quick return if possible.
	if ((*n) == 0) || ((*alpha) == real(zero)) {
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
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(j-1)
					k = kk
					for i = 1; i <= j-1; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k = k + 1
					}
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(j-1)*temp), 0))
				} else {
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
				}
				kk = kk + j
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(jx-1)
					ix = kx
					for k = kk; k <= kk+j-2; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
						ix = ix + (*incx)
					}
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(jx-1)*temp), 0))
				} else {
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
				}
				jx = jx + (*incx)
				kk = kk + j
			}
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		if (*incx) == 1 {
			for j = 1; j <= (*n); j++ {
				if x.Get(j-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(j-1)
					ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(temp*x.Get(j-1)), 0))
					k = kk + 1
					for i = j + 1; i <= (*n); i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp)
						k = k + 1
					}
				} else {
					ap.Set(kk-1, ap.GetReCmplx(kk-1))
				}
				kk = kk + (*n) - j + 1
			}
		} else {
			jx = kx
			for j = 1; j <= (*n); j++ {
				if x.Get(jx-1) != zero {
					temp = complex(*alpha, 0) * x.GetConj(jx-1)
					ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(temp*x.Get(jx-1)), 0))
					ix = jx
					for k = kk + 1; k <= kk+(*n)-j; k++ {
						ix = ix + (*incx)
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
					}
				} else {
					ap.Set(kk-1, ap.GetReCmplx(kk-1))
				}
				jx = jx + (*incx)
				kk = kk + (*n) - j + 1
			}
		}
	}
}

// Zher2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n hermitian matrix.
func Zher2(uplo mat.MatUplo, n *int, alpha *complex128, x *mat.CVector, incx *int, y *mat.CVector, incy *int, a *mat.CMatrix, lda *int) {
	var temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, kx, ky int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*incy) == 0 {
		info = 7
	} else if (*lda) < maxint(1, *n) {
		info = 9
	}
	if info != 0 {
		Xerbla([]byte("ZHER2 "), info)
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
	if uplo == Upper {
		//        Form  A  when A is stored in the upper triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.GetConj(j-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(j-1))
					for i = 1; i <= j-1; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
					}
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(j-1)*temp1+y.Get(j-1)*temp2), 0))
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.GetConj(jy-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(jx-1))
					ix = kx
					iy = ky
					for i = 1; i <= j-1; i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix = ix + (*incx)
						iy = iy + (*incy)
					}
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.GetConj(j-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(j-1))
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(j-1)*temp1+y.Get(j-1)*temp2), 0))
					for i = j + 1; i <= (*n); i++ {
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
					}
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.GetConj(jy-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(jx-1))
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
					ix = jx
					iy = jy
					for i = j + 1; i <= (*n); i++ {
						ix = ix + (*incx)
						iy = iy + (*incy)
						a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
					}
				} else {
					a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
				}
				jx = jx + (*incx)
				jy = jy + (*incy)
			}
		}
	}
}

// Zhpr2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n hermitian matrix, supplied in packed form.
func Zhpr2(uplo mat.MatUplo, n *int, alpha *complex128, x *mat.CVector, incx *int, y *mat.CVector, incy *int, ap *mat.CVector) {
	var temp1, temp2, zero complex128
	var i, info, ix, iy, j, jx, jy, k, kk, kx, ky int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if uplo != Upper && uplo != Lower {
		info = 1
	} else if (*n) < 0 {
		info = 2
	} else if (*incx) == 0 {
		info = 5
	} else if (*incy) == 0 {
		info = 7
	}
	if info != 0 {
		Xerbla([]byte("ZHPR2 "), info)
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
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.GetConj(j-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(j-1))
					k = kk
					for i = 1; i <= j-1; i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
						k = k + 1
					}
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(j-1)*temp1+y.Get(j-1)*temp2), 0))
				} else {
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
				}
				kk = kk + j
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.GetConj(jy-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(jx-1))
					ix = kx
					iy = ky
					for k = kk; k <= kk+j-2; k++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
						ix = ix + (*incx)
						iy = iy + (*incy)
					}
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
				} else {
					ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
				}
				jx = jx + (*incx)
				jy = jy + (*incy)
				kk = kk + j
			}
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		if ((*incx) == 1) && ((*incy) == 1) {
			for j = 1; j <= (*n); j++ {
				if (x.Get(j-1) != zero) || (y.Get(j-1) != zero) {
					temp1 = (*alpha) * y.GetConj(j-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(j-1))
					ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(x.Get(j-1)*temp1+y.Get(j-1)*temp2), 0))
					k = kk + 1
					for i = j + 1; i <= (*n); i++ {
						ap.Set(k-1, ap.Get(k-1)+x.Get(i-1)*temp1+y.Get(i-1)*temp2)
						k = k + 1
					}
				} else {
					ap.Set(kk-1, ap.GetReCmplx(kk-1))
				}
				kk = kk + (*n) - j + 1
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
					temp1 = (*alpha) * y.GetConj(jy-1)
					temp2 = cmplx.Conj((*alpha) * x.Get(jx-1))
					ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
					ix = jx
					iy = jy
					for k = kk + 1; k <= kk+(*n)-j; k++ {
						ix = ix + (*incx)
						iy = iy + (*incy)
						ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
					}
				} else {
					ap.Set(kk-1, ap.GetReCmplx(kk-1))
				}
				jx = jx + (*incx)
				jy = jy + (*incy)
				kk = kk + (*n) - j + 1
			}
		}
	}
}
