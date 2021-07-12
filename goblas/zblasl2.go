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
func Zgbmv(trans mat.MatTrans, m, n, kl, ku int, alpha complex128, a *mat.CMatrix, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, info, iy, j, k, kup1, lenx, leny int

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
	} else if a.Rows < (kl + ku + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, (kl + ku + 1))
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

	xiter := x.Iter(lenx)
	yiter := y.Iter(leny)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the band part of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				y.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				y.Set(iy, beta*y.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	kup1 = ku + 1
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j = 0; j < n; j++ {
			temp = alpha * x.Get(xiter[j])
			k = kup1 - j - 1
			for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp*a.Get(k+i, j))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j = 0; j < n; j++ {
			temp = zero
			k = kup1 - j - 1
			if noconj {
				for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
					temp += a.Get(k+i, j) * x.Get(xiter[i])
				}
			} else {
				for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
					temp += a.GetConj(k+i, j) * x.Get(xiter[i])
				}
			}
			y.Set(yiter[j], y.Get(yiter[j])+alpha*temp)
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
func Zgemv(trans mat.MatTrans, m, n int, alpha complex128, a *mat.CMatrix, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, iy, j, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
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

	xiter := x.Iter(lenx)
	yiter := y.Iter(leny)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				y.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				y.Set(iy, beta*y.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j = 0; j < n; j++ {
			temp = alpha * x.Get(xiter[j])
			for i = 0; i < m; i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp*a.Get(i, j))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j = 0; j < n; j++ {
			temp = zero
			if noconj {
				for i = 0; i < m; i++ {
					temp += a.Get(i, j) * x.Get(xiter[i])
				}
			} else {
				for i = 0; i < m; i++ {
					temp += a.GetConj(i, j) * x.Get(xiter[i])
				}
			}
			y.Set(yiter[j], y.Get(yiter[j])+alpha*temp)
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
func Zhbmv(uplo mat.MatUplo, n, k int, alpha complex128, a *mat.CMatrix, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j, kplus1, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
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
	xiter := x.Iter(n)
	yiter := y.Iter(n)

	//     Start the operations. In this version the elements of the array A
	//     are accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				y.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				y.Set(iy, beta*y.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = k
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			l = kplus1 - j
			for i = max(0, j-k); i < j; i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.GetConj(l+i, j) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+temp1*a.GetReCmplx(kplus1, j)+alpha*temp2)
		}
	} else {
		//        Form  y  when lower triangle of A is stored.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			y.Set(yiter[j], y.Get(yiter[j])+temp1*a.GetReCmplx(0, j))
			l = 1 - j - 1
			for i = j + 1; i < min(n, j+k+1); i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.GetConj(l+i, j) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
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
func Zhemv(uplo mat.MatUplo, n int, alpha complex128, a *mat.CMatrix, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
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
	xiter := x.Iter(n)
	yiter := y.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				y.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				y.Set(iy, beta*y.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for i = 0; i < j; i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.GetConj(i, j) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+temp1*a.GetReCmplx(j, j)+alpha*temp2)
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			y.Set(yiter[j], y.Get(yiter[j])+temp1*a.GetReCmplx(j, j))
			for i = j + 1; i < n; i++ {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.GetConj(i, j) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
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
func Zhpmv(uplo mat.MatUplo, n int, alpha complex128, ap, x *mat.CVector, beta complex128, y *mat.CVector) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j, k, kk int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
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
	xiter := x.Iter(n)
	yiter := y.Iter(n)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				y.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				y.Set(iy, beta*y.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when AP contains the upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*ap.Get(k))
				temp2 += ap.GetConj(k) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+temp1*ap.GetReCmplx(kk+j)+alpha*temp2)
			kk += j + 1
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			y.Set(yiter[j], y.Get(yiter[j])+temp1*ap.GetReCmplx(kk))
			for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
				y.Set(yiter[i], y.Get(yiter[i])+temp1*ap.Get(k))
				temp2 += ap.GetConj(k) * x.Get(xiter[i])
			}
			y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
			kk += (n - j)
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
func Ztbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.CMatrix, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, kplus1, l int

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
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//         Form  x := A*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					l = kplus1 - j - 1
					for i = max(0, j-k); i < j; i++ {
						x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*a.Get(kplus1-1, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					l = 1 - j - 1
					for i = min(n-1, j+k); i >= j+1; i-- {
						x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*a.Get(0, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kplus1 = k
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				l = kplus1 - j
				if noconj {
					if nounit {
						temp *= a.Get(kplus1, j)
					}
					for i = j - 1; i >= max(0, j-k); i-- {
						temp += a.Get(l+i, j) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(kplus1, j)
					}
					for i = j - 1; i >= max(0, j-k); i-- {
						temp += a.GetConj(l+i, j) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				l = 1 - j - 1
				if noconj {
					if nounit {
						temp *= a.Get(0, j)
					}
					for i = j + 1; i < min(n, j+k+1); i++ {
						temp += a.Get(l+i, j) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(0, j)
					}
					for i = j + 1; i < min(n, j+k+1); i++ {
						temp += a.GetConj(l+i, j) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
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
func Ztbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.CMatrix, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, kplus1, l int

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
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kplus1 = k
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					l = kplus1 - j
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/a.Get(kplus1, j))
					}
					temp = x.Get(xiter[j])
					for i = j - 1; i >= max(0, j-k); i-- {
						x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					l = 1 - j - 1
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/a.Get(0, j))
					}
					temp = x.Get(xiter[j])
					for i = j + 1; i < min(n, j+k+1); i++ {
						x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kplus1 = k
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				l = kplus1 - j
				if noconj {
					for i = max(0, j-k); i < j; i++ {
						temp -= a.Get(l+i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(kplus1, j)
					}
				} else {
					for i = max(0, j-k); i < j; i++ {
						temp -= a.GetConj(l+i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(kplus1, j)
					}
				}
				x.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				l = -j
				if noconj {
					for i = min(n-1, j+k); i >= j+1; i-- {
						temp -= a.Get(l+i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(0, j)
					}
				} else {
					for i = min(n-1, j+k); i >= j+1; i-- {
						temp -= a.GetConj(l+i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(0, j)
					}
				}
				x.Set(xiter[j], temp)
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
func Ztpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, k, kk int

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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x:= A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						x.Set(xiter[i], x.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*ap.Get(kk+j))
					}
				}
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						x.Set(xiter[i], x.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*ap.Get(kk-n+j))
					}
				}
				kk -= (n - j)
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kk = (n*(n+1))/2 - 1
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= ap.Get(kk)
					}
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						temp += ap.Get(k) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk)
					}
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						temp += ap.GetConj(k) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
				kk -= j + 1
			}
		} else {
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= ap.Get(kk)
					}
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						temp += ap.Get(k) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk)
					}
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						temp += ap.GetConj(k) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
				kk += (n - j)
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
func Ztpsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, k, kk int

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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kk = (n*(n+1))/2 - 1
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/ap.Get(kk))
					}
					temp = x.Get(xiter[j])
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						x.Set(xiter[i], x.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk -= j + 1
			}
		} else {
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/ap.Get(kk))
					}
					temp = x.Get(xiter[j])
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						x.Set(xiter[i], x.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk += (n - j)
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				if noconj {
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						temp -= ap.Get(k) * x.Get(xiter[i])
					}
					if nounit {
						temp /= ap.Get(kk + j)
					}
				} else {
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						temp -= ap.GetConj(k) * x.Get(xiter[i])
					}
					if nounit {
						temp /= ap.GetConj(kk + j)
					}
				}
				x.Set(xiter[j], temp)
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				if noconj {
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						temp -= ap.Get(k) * x.Get(xiter[i])
					}
					if nounit {
						temp /= ap.Get(kk - n + j)
					}
				} else {
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						temp -= ap.GetConj(k) * x.Get(xiter[i])
					}
					if nounit {
						temp /= ap.GetConj(kk - n + j)
					}
				}
				x.Set(xiter[j], temp)
				kk -= (n - j)
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
func Ztrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.CMatrix, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j int

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
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					for i = 0; i < j; i++ {
						x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					temp = x.Get(xiter[j])
					for i = n - 1; i >= j+1; i-- {
						x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= a.Get(j, j)
					}
					for i = j - 1; i >= 0; i-- {
						temp += a.Get(i, j) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(j, j)
					}
					for i = j - 1; i >= 0; i-- {
						temp += a.GetConj(i, j) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= a.Get(j, j)
					}
					for i = j + 1; i < n; i++ {
						temp += a.Get(i, j) * x.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(j, j)
					}
					for i = j + 1; i < n; i++ {
						temp += a.GetConj(i, j) * x.Get(xiter[i])
					}
				}
				x.Set(xiter[j], temp)
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
func Ztrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.CMatrix, x *mat.CVector) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j int

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
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				if x.Get(xiter[j]) != zero {
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/a.Get(j, j))
					}
					temp = x.Get(xiter[j])
					for i = j - 1; i >= 0; i-- {
						x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if x.Get(xiter[j]) != zero {
					if nounit {
						x.Set(xiter[j], x.Get(xiter[j])/a.Get(j, j))
					}
					temp = x.Get(xiter[j])
					for i = j + 1; i < n; i++ {
						x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = x.Get(xiter[j])
				if noconj {
					for i = 0; i < j; i++ {
						temp -= a.Get(i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(j, j)
					}
				} else {
					for i = 0; i < j; i++ {
						temp -= a.GetConj(i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(j, j)
					}
				}
				x.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = x.Get(xiter[j])
				if noconj {
					for i = n - 1; i >= j+1; i-- {
						temp -= a.Get(i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(j, j)
					}
				} else {
					for i = n - 1; i >= j+1; i-- {
						temp -= a.GetConj(i, j) * x.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(j, j)
					}
				}
				x.Set(xiter[j], temp)
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
func Zgerc(m, n int, alpha complex128, x *mat.CVector, y *mat.CVector, a *mat.CMatrix) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
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
	xiter := x.Iter(m)
	yiter := y.Iter(n)

	for j = 0; j < n; j++ {
		if y.Get(yiter[j]) != zero {
			temp = alpha * y.GetConj(yiter[j])
			for i = 0; i < m; i++ {
				a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
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
func Zgeru(m, n int, alpha complex128, x *mat.CVector, y *mat.CVector, a *mat.CMatrix) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
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
	xiter := x.Iter(m)
	yiter := y.Iter(n)

	for j = 0; j < n; j++ {
		if y.Get(yiter[j]) != zero {
			temp = alpha * y.Get(yiter[j])
			for i = 0; i < m; i++ {
				a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
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
func Zher(uplo mat.MatUplo, n int, alpha float64, x *mat.CVector, a *mat.CMatrix) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in upper triangle.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				for i = 0; i < j; i++ {
					a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
				}
				a.Set(j, j, a.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp), 0))
			} else {
				a.Set(j, j, a.GetReCmplx(j, j))
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				a.Set(j, j, a.GetReCmplx(j, j)+complex(real(temp*x.Get(xiter[j])), 0))
				for i = j + 1; i < n; i++ {
					a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
				}
			} else {
				a.Set(j, j, a.GetReCmplx(j, j))
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
func Zhpr(uplo mat.MatUplo, n int, alpha float64, x *mat.CVector, ap *mat.CVector) (err error) {
	var temp, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
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
	xiter := x.Iter(n)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
					ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp)
				}
				ap.Set(kk+j, ap.GetReCmplx(kk+j)+complex(real(x.Get(xiter[j])*temp), 0))
			} else {
				ap.Set(kk+j, ap.GetReCmplx(kk+j))
			}
			kk += j + 1
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 1; j <= n; j++ {
			if x.Get(xiter[j-1]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j-1])
				ap.Set(kk, ap.GetReCmplx(kk)+complex(real(temp*x.Get(xiter[j-1])), 0))
				for i, k = j, kk+2; k <= kk+n-j+1; i, k = i+1, k+1 {
					ap.Set(k-1, ap.Get(k-1)+x.Get(xiter[i])*temp)
				}
			} else {
				ap.Set(kk, ap.GetReCmplx(kk))
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
func Zher2(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, y *mat.CVector, a *mat.CMatrix) (err error) {
	var temp1, temp2, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
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
	xiter := x.Iter(n)
	yiter := y.Iter(n)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in the upper triangle.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				for i = 0; i < j; i++ {
					a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
				a.Set(j, j, a.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
			} else {
				a.Set(j, j, a.GetReCmplx(j, j))
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				a.Set(j, j, a.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
				for i = j + 1; i < n; i++ {
					a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			} else {
				a.Set(j, j, a.GetReCmplx(j, j))
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
func Zhpr2(uplo mat.MatUplo, n int, alpha complex128, x *mat.CVector, y *mat.CVector, ap *mat.CVector) (err error) {
	var temp1, temp2, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
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
	xiter := x.Iter(n)
	yiter := y.Iter(n)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
					ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
				ap.Set(kk+j, ap.GetReCmplx(kk+j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
			} else {
				ap.Set(kk+j, ap.GetReCmplx(kk+j))
			}
			kk += j + 1
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				ap.Set(kk, ap.GetReCmplx(kk)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
				for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
					ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			} else {
				ap.Set(kk, ap.GetReCmplx(kk))
			}
			kk += n - j
		}
	}

	return
}
