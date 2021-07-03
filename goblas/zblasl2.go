package goblas

import (
	"fmt"
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
func Zgbmv(trans mat.MatTrans, m, n, kl, ku int, alpha complex128, a *mat.CMatrix, lda int, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, info, ix, iy, j, jx, jy, k, kup1, kx, ky, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl invalid: %v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku invalid: %v", ku)
	} else if lda < (kl + ku + 1) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if info != 0 {
		Xerbla([]byte("Zgbmv"), info)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (lenx-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (leny-1)*incy
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the band part of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		iy = ky
		if beta == zero {
			for i = 1; i <= leny; i, iy = i+1, iy+incy {
				y.Set(iy-1, zero)
			}
		} else {
			for i = 1; i <= leny; i, iy = i+1, iy+incy {
				y.Set(iy-1, beta*y.Get(iy-1))
			}
		}
	}
	if alpha == zero {
		return
	}
	kup1 = ku + 1
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			temp = alpha * x.Get(jx-1)
			k = kup1 - j
			for i, iy = max(1, j-ku), ky; i <= min(m, j+kl); i, iy = i+1, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp*a.Get(k+i-1, j-1))
			}
			if j > ku {
				ky += incy
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j, jy = 1, ky; j <= n; j, jy = j+1, jy+incy {
			temp = zero
			ix = kx
			k = kup1 - j
			if noconj {
				for i = max(1, j-ku); i <= min(m, j+kl); i, ix = i+1, ix+incx {
					temp += a.Get(k+i-1, j-1) * x.Get(ix-1)
				}
			} else {
				for i = max(1, j-ku); i <= min(m, j+kl); i, ix = i+1, ix+incx {
					temp += a.GetConj(k+i-1, j-1) * x.Get(ix-1)
				}
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp)
			if j > ku {
				kx += incx
			}
		}
	}

	return
}

// Zgemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//
//    y := alpha*A**H*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func Zgemv(trans mat.MatTrans, m, n int, alpha complex128, a *mat.CMatrix, lda int, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, ix, iy, j, jx, jy, kx, ky, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, m) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if err != nil {
		Xerbla2([]byte("Zgemv"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (lenx-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (leny-1)*incy
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		iy = ky
		if beta == zero {
			for i = 1; i <= leny; i, iy = i+1, iy+incy {
				y.Set(iy-1, zero)
			}
		} else {
			for i = 1; i <= leny; i, iy = i+1, iy+incy {
				y.Set(iy-1, beta*y.Get(iy-1))
			}
		}
	}
	if alpha == zero {
		return
	}
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			temp = alpha * x.Get(jx-1)
			for i, iy = 1, ky; i <= m; i, iy = i+1, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp*a.Get(i-1, j-1))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j, jy = 1, ky; j <= n; j, jy = j+1, jy+incy {
			temp = zero
			ix = kx
			if noconj {
				for i = 1; i <= m; i, ix = i+1, ix+incx {
					temp += a.Get(i-1, j-1) * x.Get(ix-1)
				}
			} else {
				for i = 1; i <= m; i, ix = i+1, ix+incx {
					temp += a.GetConj(i-1, j-1) * x.Get(ix-1)
				}
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp)
		}
	}

	return
}

// Zhbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian band matrix, with k super-diagonals.
func Zhbmv(uplo mat.MatUplo, n, k int, alpha complex128, a *mat.CMatrix, lda int, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, ix, iy, j, jx, jy, kplus1, kx, ky, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < (k + 1) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if err != nil {
		Xerbla2([]byte("Zhbmv"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (n-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (n-1)*incy
	}

	//     Start the operations. In this version the elements of the array A
	//     are accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		iy = ky
		if beta == zero {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, zero)
			}
		} else {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, beta*y.Get(iy-1))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = k + 1
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			l = kplus1 - j
			for i, ix, iy = max(1, j-k), kx, ky; i <= j-1; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
				temp2 += a.GetConj(l+i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(kplus1-1, j-1)+alpha*temp2)
			if j > k {
				kx += incx
				ky += incy
			}
		}
	} else {
		//        Form  y  when lower triangle of A is stored.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(0, j-1))
			l = 1 - j
			for i, ix, iy = j+1, jx+incx, jy+incy; i <= min(n, j+k); i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
				temp2 += a.GetConj(l+i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
		}
	}

	return
}

// Zhemv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix.
func Zhemv(uplo mat.MatUplo, n int, alpha complex128, a *mat.CMatrix, lda int, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, ix, iy, j, jx, jy, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, n) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if err != nil {
		Xerbla2([]byte("Zhemv"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (n-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (n-1)*incy
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		// if incy == 1 {
		iy = ky
		if beta == zero {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, zero)
			}
		} else {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, beta*y.Get(iy-1))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			for i, ix, iy = 1, kx, ky; i <= j-1; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
				temp2 += a.GetConj(i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(j-1, j-1)+alpha*temp2)
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*a.GetReCmplx(j-1, j-1))
			for i, ix, iy = j+1, jx+incx, jy+incy; i <= n; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
				temp2 += a.GetConj(i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
		}
	}

	return
}

// Zhpmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix, supplied in packed form.
func Zhpmv(uplo mat.MatUplo, n int, alpha complex128, ap, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, ix, iy, j, jx, jy, k, kk, kx, ky int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if err != nil {
		Xerbla2([]byte("Zhpmv"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (n-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (n-1)*incy
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	//
	//     First form  y := beta*y.
	if beta != one {
		iy = ky
		if beta == zero {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, zero)
			}
		} else {
			for i = 1; i <= n; i, iy = i+1, iy+incy {
				y.Set(iy-1, beta*y.Get(iy-1))
			}
		}
	}
	if alpha == zero {
		return
	}
	kk = 1
	if uplo == Upper {
		//        Form  y  when AP contains the upper triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			for k, ix, iy = kk, kx, ky; k <= kk+j-2; k, ix, iy = k+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
				temp2 += ap.GetConj(k-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*ap.GetReCmplx(kk+j-1-1)+alpha*temp2)
			kk += j
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*ap.GetReCmplx(kk-1))
			for k, ix, iy = kk+1, jx+incx, jy+incy; k <= kk+n-j; k, ix, iy = k+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
				temp2 += ap.GetConj(k-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
			kk += (n - j + 1)
		}
	}

	return
}

// Ztbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func Ztbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.CMatrix, lda int, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, ix, j, jx, kplus1, kx, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < (k + 1) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztbmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX   too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//         Form  x := A*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					l = kplus1 - j
					for i, ix = max(1, j-k), kx; i <= j-1; i, ix = i+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)+temp*a.Get(l+i-1, j-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*a.Get(kplus1-1, j-1))
					}
				}
				if j > k {
					kx = kx + incx
				}
			}
		} else {
			kx = kx + (n-1)*incx
			jx = kx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					l = 1 - j
					for i, ix = min(n, j+k), kx; i >= j+1; i, ix = i-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)+temp*a.Get(l+i-1, j-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*a.Get(0, j-1))
					}
				}
				if (n - j) >= k {
					kx -= incx
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kplus1 = k + 1
			kx = kx + (n-1)*incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				kx = kx - incx
				ix = kx
				l = kplus1 - j
				if noconj {
					if nounit {
						temp *= a.Get(kplus1-1, j-1)
					}
					for i = j - 1; i >= max(1, j-k); i, ix = i-1, ix-incx {
						temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= a.GetConj(kplus1-1, j-1)
					}
					for i = j - 1; i >= max(1, j-k); i, ix = i-1, ix-incx {
						temp += a.GetConj(l+i-1, j-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				kx = kx + incx
				ix = kx
				l = 1 - j
				if noconj {
					if nounit {
						temp *= a.Get(0, j-1)
					}
					for i = j + 1; i <= min(n, j+k); i, ix = i+1, ix+incx {
						temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= a.GetConj(0, j-1)
					}
					for i = j + 1; i <= min(n, j+k); i, ix = i+1, ix+incx {
						temp += a.GetConj(l+i-1, j-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
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
func Ztbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.CMatrix, lda int, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, ix, j, jx, kplus1, kx, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < (k + 1) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztbsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kplus1 = k + 1
			kx = kx + (n-1)*incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				kx -= incx
				if x.Get(jx-1) != zero {
					l = kplus1 - j
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/a.Get(kplus1-1, j-1))
					}
					temp = x.Get(jx - 1)
					for i, ix = j-1, kx; i >= max(1, j-k); i, ix = i-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
					}
				}
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				kx += incx
				if x.Get(jx-1) != zero {
					l = 1 - j
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/a.Get(0, j-1))
					}
					temp = x.Get(jx - 1)
					for i, ix = j+1, kx; i <= min(n, j+k); i, ix = i+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)-temp*a.Get(l+i-1, j-1))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				ix = kx
				l = kplus1 - j
				if noconj {
					for i = max(1, j-k); i <= j-1; i, ix = i+1, ix+incx {
						temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.Get(kplus1-1, j-1)
					}
				} else {
					for i = max(1, j-k); i <= j-1; i, ix = i+1, ix+incx {
						temp -= a.GetConj(l+i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.GetConj(kplus1-1, j-1)
					}
				}
				x.Set(jx-1, temp)
				if j > k {
					kx += incx
				}
			}
		} else {
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				ix = kx
				l = 1 - j
				if noconj {
					for i = min(n, j+k); i >= j+1; i, ix = i-1, ix-incx {
						temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.Get(0, j-1)
					}
				} else {
					for i = min(n, j+k); i >= j+1; i, ix = i-1, ix-incx {
						temp -= a.GetConj(l+i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.GetConj(0, j-1)
					}
				}
				x.Set(jx-1, temp)
				if (n - j) >= k {
					kx -= incx
				}
			}
		}
	}

	return
}

// Ztpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func Ztpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztpmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x:= A*x.
		if uplo == Upper {
			kk = 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					for k, ix = kk, kx; k <= kk+j-2; k, ix = k+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*ap.Get(kk+j-1-1))
					}
				}
				kk += j
			}
		} else {
			kk = (n * (n + 1)) / 2
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					for k, ix = kk, kx; k >= kk-(n-(j+1)); k, ix = k-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)+temp*ap.Get(k-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*ap.Get(kk-n+j-1))
					}
				}
				kk -= (n - j + 1)
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kk = (n * (n + 1)) / 2
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				ix = jx - incx
				if noconj {
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					for k = kk - 1; k >= kk-j+1; k, ix = k-1, ix-incx {
						temp += ap.Get(k-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk - 1)
					}
					for k = kk - 1; k >= kk-j+1; k, ix = k-1, ix-incx {
						temp += ap.GetConj(k-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
				kk -= j
			}
		} else {
			kk = 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				ix = jx + incx
				if noconj {
					if nounit {
						temp *= ap.Get(kk - 1)
					}
					for k = kk + 1; k <= kk+n-j; k, ix = k+1, ix+incx {
						temp += ap.Get(k-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk - 1)
					}
					for k = kk + 1; k <= kk+n-j; k, ix = k+1, ix+incx {
						temp += ap.GetConj(k-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
				kk += (n - j + 1)
			}
		}
	}

	return
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
func Ztpsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztpsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kk = (n * (n + 1)) / 2
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				if x.Get(jx-1) != zero {
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/ap.Get(kk-1))
					}
					temp = x.Get(jx - 1)
					for k, ix = kk-1, jx-incx; k >= kk-j+1; k, ix = k-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
					}
				}
				kk -= j
			}
		} else {
			kk = 1
			jx = kx
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				if x.Get(jx-1) != zero {
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/ap.Get(kk-1))
					}
					temp = x.Get(jx - 1)
					for k, ix = kk+1, jx+incx; k <= kk+n-j; k, ix = k+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)-temp*ap.Get(k-1))
					}
				}
				kk += (n - j + 1)
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kk = 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				ix = kx
				if noconj {
					for k = kk; k <= kk+j-2; k, ix = k+1, ix+incx {
						temp -= ap.Get(k-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= ap.Get(kk + j - 1 - 1)
					}
				} else {
					for k = kk; k <= kk+j-2; k, ix = k+1, ix+incx {
						temp -= ap.GetConj(k-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= ap.GetConj(kk + j - 1 - 1)
					}
				}
				x.Set(jx-1, temp)
				kk += j
			}
		} else {
			kk = (n * (n + 1)) / 2
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				ix = kx
				if noconj {
					for k = kk; k >= kk-(n-(j+1)); k, ix = k-1, ix-incx {
						temp -= ap.Get(k-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= ap.Get(kk - n + j - 1)
					}
				} else {
					for k = kk; k >= kk-(n-(j+1)); k, ix = k-1, ix-incx {
						temp -= ap.GetConj(k-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= ap.GetConj(kk - n + j - 1)
					}
				}
				x.Set(jx-1, temp)
				kk -= (n - j + 1)
			}
		}
	}

	return
}

// Ztrmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func Ztrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.CMatrix, lda int, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, n) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztrmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := A*x.
		if uplo == Upper {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					for i, ix = 1, kx; i <= j-1; i, ix = i+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)+temp*a.Get(i-1, j-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
					}
				}
			}
		} else {
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				if x.Get(jx-1) != zero {
					temp = x.Get(jx - 1)
					for i, ix = n, kx; i >= j+1; i, ix = i-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)+temp*a.Get(i-1, j-1))
					}
					if nounit {
						x.Set(jx-1, x.Get(jx-1)*a.Get(j-1, j-1))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				ix = jx - incx
				if noconj {
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j - 1; i >= 1; i, ix = i-1, ix-incx {
						temp += a.Get(i-1, j-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= a.GetConj(j-1, j-1)
					}
					for i = j - 1; i >= 1; i, ix = i-1, ix-incx {
						temp += a.GetConj(i-1, j-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				ix = jx + incx
				if noconj {
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = j + 1; i <= n; i, ix = i+1, ix+incx {
						temp += a.Get(i-1, j-1) * x.Get(ix-1)
					}
				} else {
					if nounit {
						temp *= a.GetConj(j-1, j-1)
					}
					for i = j + 1; i <= n; i, ix = i+1, ix+incx {
						temp += a.GetConj(i-1, j-1) * x.Get(ix-1)
					}
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
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
func Ztrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.CMatrix, lda int, x *mat.CVector, incx int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, n) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Ztrsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				if x.Get(jx-1) != zero {
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/a.Get(j-1, j-1))
					}
					temp = x.Get(jx - 1)
					for i, ix = j-1, jx-incx; i >= 1; i, ix = i-1, ix-incx {
						x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
					}
				}
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				if x.Get(jx-1) != zero {
					if nounit {
						x.Set(jx-1, x.Get(jx-1)/a.Get(j-1, j-1))
					}
					temp = x.Get(jx - 1)
					for i, ix = j+1, jx+incx; i <= n; i, ix = i+1, ix+incx {
						x.Set(ix-1, x.Get(ix-1)-temp*a.Get(i-1, j-1))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				ix = kx
				temp = x.Get(jx - 1)
				if noconj {
					for i = 1; i <= j-1; i, ix = i+1, ix+incx {
						temp -= a.Get(i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
				} else {
					for i = 1; i <= j-1; i, ix = i+1, ix+incx {
						temp -= a.GetConj(i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.GetConj(j-1, j-1)
					}
				}
				x.Set(jx-1, temp)
			}
		} else {
			kx = kx + (n-1)*incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				ix = kx
				temp = x.Get(jx - 1)
				if noconj {
					for i = n; i >= j+1; i, ix = i-1, ix-incx {
						temp -= a.Get(i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.Get(j-1, j-1)
					}
				} else {
					for i = n; i >= j+1; i, ix = i-1, ix-incx {
						temp -= a.GetConj(i-1, j-1) * x.Get(ix-1)
					}
					if nounit {
						temp /= a.GetConj(j-1, j-1)
					}
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
}

// Zgerc performs the rank 1 operation
//
//    A := alpha*x*y**H + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Zgerc(m, n int, alpha complex128, x *mat.CVector, incx int, y *mat.CVector, incy int, a *mat.CMatrix, lda int) (err error) {
	var temp, zero complex128
	var i, ix, j, jy, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	} else if lda < max(1, m) {
		err = fmt.Errorf("lda invalid: %v", lda)
	}
	if err != nil {
		Xerbla2([]byte("Zgerc"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (alpha == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if incy > 0 {
		jy = 1
	} else {
		jy = 1 - (n-1)*incy
	}
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (m-1)*incx
	}
	for j = 1; j <= n; j, jy = j+1, jy+incy {
		if y.Get(jy-1) != zero {
			temp = alpha * y.GetConj(jy-1)
			for i, ix = 1, kx; i <= m; i, ix = i+1, ix+incx {
				a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
			}
		}
	}

	return
}

// Zgeru performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Zgeru(m, n int, alpha complex128, x *mat.CVector, incx int, y *mat.CVector, incy int, a *mat.CMatrix, lda int) (err error) {
	var temp, zero complex128
	var i, ix, j, jy, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	} else if lda < max(1, m) {
		err = fmt.Errorf("lda invalid: %v", lda)
	}
	if err != nil {
		Xerbla2([]byte("Zgeru"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (alpha == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if incy > 0 {
		jy = 1
	} else {
		jy = 1 - (n-1)*incy
	}
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (m-1)*incx
	}
	for j = 1; j <= n; j, jy = j+1, jy+incy {
		if y.Get(jy-1) != zero {
			temp = alpha * y.Get(jy-1)
			for i, ix = 1, kx; i <= m; i, ix = i+1, ix+incx {
				a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
			}
		}
	}

	return
}

// Zher performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix.
func Zher(uplo mat.MatUplo, n int, alpha float64, x *mat.CVector, incx int, a *mat.CMatrix, lda int) (err error) {
	var temp, zero complex128
	var i, ix, j, jx, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if lda < max(1, n) {
		err = fmt.Errorf("lda invalid: %v", lda)
	}
	if err != nil {
		Xerbla2([]byte("Zher"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == real(zero)) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in upper triangle.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = complex(alpha, 0) * x.GetConj(jx-1)
				for i, ix = 1, kx; i <= j-1; i, ix = i+1, ix+incx {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
				}
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp), 0))
			} else {
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = complex(alpha, 0) * x.GetConj(jx-1)
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(temp*x.Get(jx-1)), 0))
				for i, ix = j+1, jx+incx; i <= n; i, ix = i+1, ix+incx {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
				}
			} else {
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
			}
		}
	}

	return
}

// Zhpr performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix, supplied in packed form.
func Zhpr(uplo mat.MatUplo, n int, alpha float64, x *mat.CVector, incx int, ap *mat.CVector) (err error) {
	var temp, zero complex128
	var ix, j, jx, k, kk, kx int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Zhpr"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == real(zero)) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	if incx <= 0 {
		kx = 1 - (n-1)*incx
	} else if incx != 1 {
		kx = 1
	}
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = complex(alpha, 0) * x.GetConj(jx-1)
				for k, ix = kk, kx; k <= kk+j-2; k, ix = k+1, ix+incx {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
				}
				ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(jx-1)*temp), 0))
			} else {
				ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
			}
			kk += j
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = complex(alpha, 0) * x.GetConj(jx-1)
				ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(temp*x.Get(jx-1)), 0))
				for k, ix = kk+1, jx+incx; k <= kk+n-j; k, ix = k+1, ix+incx {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
				}
			} else {
				ap.Set(kk-1, ap.GetReCmplx(kk-1))
			}
			kk += (n - j + 1)
		}
	}

	return
}

// Zher2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n hermitian matrix.
func Zher2(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, incx int, y *mat.CVector, incy int, a *mat.CMatrix, lda int) (err error) {
	var temp1, temp2, zero complex128
	var i, ix, iy, j, jx, jy, kx, ky int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	} else if lda < max(1, n) {
		err = fmt.Errorf("lda invalid: %v", lda)
	}
	if err != nil {
		Xerbla2([]byte("Zher2"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	if (incx != 1) || (incy != 1) {
		if incx > 0 {
			kx = 1
		} else {
			kx = 1 - (n-1)*incx
		}
		if incy > 0 {
			ky = 1
		} else {
			ky = 1 - (n-1)*incy
		}
		jx = kx
		jy = ky
	}
	if incx == 1 && incy == 1 {
		kx++
		ky++
		jx++
		jy++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in the upper triangle.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.GetConj(jy-1)
				temp2 = cmplx.Conj(alpha * x.Get(jx-1))
				for i, ix, iy = 1, kx, ky; i <= j-1; i, ix, iy = i+1, ix+incx, iy+incy {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
			} else {
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.GetConj(jy-1)
				temp2 = cmplx.Conj(alpha * x.Get(jx-1))
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
				for i, ix, iy = j+1, jx+incx, jy+incy; i <= n; i, ix, iy = i+1, ix+incx, iy+incy {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			} else {
				a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
			}
		}
	}

	return
}

// Zhpr2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n hermitian matrix, supplied in packed form.
func Zhpr2(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, incx int, y *mat.CVector, incy int, ap *mat.CVector) (err error) {
	var temp1, temp2, zero complex128
	var ix, iy, j, jx, jy, k, kk, kx, ky int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	} else if incy == 0 {
		err = fmt.Errorf("incy invalid: %v", incy)
	}
	if err != nil {
		Xerbla2([]byte("Zhpr2"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	if (incx != 1) || (incy != 1) {
		if incx > 0 {
			kx = 1
		} else {
			kx = 1 - (n-1)*incx
		}
		if incy > 0 {
			ky = 1
		} else {
			ky = 1 - (n-1)*incy
		}
		jx = kx
		jy = ky
	}
	if incx == 1 && incy == 1 {
		kx++
		ky++
		jx++
		jy++
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.GetConj(jy-1)
				temp2 = cmplx.Conj(alpha * x.Get(jx-1))
				for k, ix, iy = kk, kx, ky; k <= kk+j-2; k, ix, iy = k+1, ix+incx, iy+incy {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
				ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
			} else {
				ap.Set(kk+j-1-1, ap.GetReCmplx(kk+j-1-1))
			}
			kk += j
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.GetConj(jy-1)
				temp2 = cmplx.Conj(alpha * x.Get(jx-1))
				ap.Set(kk-1, ap.GetReCmplx(kk-1)+complex(real(x.Get(jx-1)*temp1+y.Get(jy-1)*temp2), 0))
				for k, ix, iy = kk+1, jx+incx, jy+incy; k <= kk+n-j; k, ix, iy = k+1, ix+incx, iy+incy {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			} else {
				ap.Set(kk-1, ap.GetReCmplx(kk-1))
			}
			kk += n - j + 1
		}
	}

	return
}
