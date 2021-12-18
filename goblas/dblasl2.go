package goblas

import (
	"github.com/whipstein/golinalg/mat"
)

// Dgbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func Dgbmv(trans mat.MatTrans, m, n, kl, ku int, alpha float64, a *mat.Matrix, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	// var one, temp, zero float64
	// var i, j, k, kup1, lenx, leny int

	// one = 1.0
	// zero = 0.0

	// //     Test the input parameters.
	// if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if m < 0 {
	// 	err = fmt.Errorf("m invalid: %v", m)
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if kl < 0 {
	// 	err = fmt.Errorf("kl invalid: %v", kl)
	// } else if ku < 0 {
	// 	err = fmt.Errorf("ku invalid: %v", ku)
	// } else if a.Rows < (kl + ku + 1) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, kl+ku+1)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dgbmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if m == 0 || n == 0 || (alpha == zero && beta == one) {
	// 	return
	// }

	// //     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	// //     up the start points in  X  and  Y.
	// if trans == mat.NoTrans {
	// 	lenx = n
	// 	leny = m
	// } else {
	// 	lenx = m
	// 	leny = n
	// }
	// xiter := x.Iter(lenx, incx)
	// yiter := y.Iter(leny, incy)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through the band part of A.
	// //
	// //     First form  y := beta*y.
	// if beta != one {
	// 	if beta == zero {
	// 		for _, i = range y.Iter(leny, incx) {
	// 			y.Set(i, zero)
	// 		}
	// 	} else {
	// 		for _, i = range y.Iter(leny, incy) {
	// 			y.Set(i, beta*y.Get(i))
	// 		}
	// 	}
	// }
	// if alpha == zero {
	// 	return
	// }
	// kup1 = ku + 1
	// if trans == mat.NoTrans {
	// 	//        Form  y := alpha*A*x + y.
	// 	for j = 0; j < n; j++ {
	// 		temp = alpha * x.Get(xiter[j])
	// 		k = kup1 - (j + 1)
	// 		for i = max(0, j-ku); i < min(m, (j+1)+kl); i++ {
	// 			y.Set(yiter[i], y.Get(yiter[i])+temp*a.Get(k+i, j))
	// 		}
	// 	}
	// } else {
	// 	//        Form  y := alpha*A**T*x + y.
	// 	for j = 0; j < n; j++ {
	// 		temp = zero
	// 		k = kup1 - (j + 1)
	// 		for i = max(0, j-ku); i < min(m, (j+1)+kl); i++ {
	// 			temp += a.Get(k+i, j) * x.Get(xiter[i])
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+alpha*temp)
	// 	}
	// }

	return y.Gbmv(trans, m, n, kl, ku, alpha, a, x, incx, beta, incy)
}

// Dgemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func Dgemv(trans mat.MatTrans, m, n int, alpha float64, a *mat.Matrix, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	// var one, temp, zero float64
	// var i, ix, iy, j, jx, jy, lenx, leny int

	// one = 1.0
	// zero = 0.0

	// //     Test the input parameters.
	// if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if m < 0 {
	// 	err = fmt.Errorf("m invalid: %v", m)
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, m) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dgemv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if m == 0 || n == 0 || (alpha == zero && beta == one) {
	// 	return
	// }

	// //     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	// //     up the start points in  X  and  Y.
	// if trans == mat.NoTrans {
	// 	lenx = n
	// 	leny = m
	// } else {
	// 	lenx = m
	// 	leny = n
	// }

	// xiter := x.Iter(lenx, incx)
	// yiter := y.Iter(leny, incy)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through A.
	// //
	// //     First form  y := beta*y.
	// if beta != one {
	// 	if beta == zero {
	// 		for _, iy = range yiter {
	// 			y.Set(iy, zero)
	// 		}
	// 	} else {
	// 		for _, iy = range yiter {
	// 			y.Set(iy, beta*y.Get(iy))
	// 		}
	// 	}
	// }
	// if alpha == zero {
	// 	return
	// }
	// if trans == mat.NoTrans {
	// 	//        Form  y := alpha*A*x + y.
	// 	for j, jx = range xiter {
	// 		temp = alpha * x.Get(jx)
	// 		for i, iy = range yiter {
	// 			y.Set(iy, y.Get(iy)+temp*a.Get(i, j))
	// 		}
	// 	}
	// } else {
	// 	for j, jy = range yiter {
	// 		temp = zero
	// 		for i, ix = range xiter {
	// 			temp += a.Get(i, j) * x.Get(ix)
	// 		}
	// 		y.Set(jy, y.Get(jy)+alpha*temp)
	// 	}
	// }

	return y.Gemv(trans, m, n, alpha, a, x, incx, beta, incy)
}

// Dsbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric band matrix, with k super-diagonals.
func Dsbmv(uplo mat.MatUplo, n, k int, alpha float64, a *mat.Matrix, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	// var one, temp1, temp2, zero float64
	// var i, j, kplus1, l int

	// one = 1.0
	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if k < 0 {
	// 	err = fmt.Errorf("k invalid: %v", k)
	// } else if a.Rows < (k + 1) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dsbmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || ((alpha == zero) && (beta == one)) {
	// 	return
	// }

	// xiter := x.Iter(n, incx)
	// yiter := y.Iter(n, incy)

	// //     Start the operations. In this version the elements of the array A
	// //     are accessed sequentially with one pass through A.
	// //
	// //     First form  y := beta*y.
	// if beta != one {
	// 	if beta == zero {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, zero)
	// 		}
	// 	} else {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, beta*y.Get(i))
	// 		}
	// 	}
	// }
	// if alpha == zero {
	// 	return
	// }
	// if uplo == mat.Upper {
	// 	//        Form  y  when upper triangle of A is stored.
	// 	kplus1 = k + 1
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		l = kplus1 - (j + 1)
	// 		for i = max(0, j-k); i < j; i++ {
	// 			y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(l+i, j))
	// 			temp2 += a.Get(l+i, j) * x.Get(xiter[i])
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*a.Get(kplus1-1, j)+alpha*temp2)
	// 	}
	// } else {
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*a.Get(0, j))
	// 		l = 1 - (j + 1)
	// 		for i = j + 1; i < min(n, j+1+k); i++ {
	// 			y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(l+i, j))
	// 			temp2 += a.Get(l+i, j) * x.Get(xiter[i])
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
	// 	}
	// }

	return y.Sbmv(uplo, n, k, alpha, a, x, incx, beta, incy)
}

// Dspmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix, supplied in packed form.
func Dspmv(uplo mat.MatUplo, n int, alpha float64, ap, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	// var one, temp1, temp2, zero float64
	// var i, ix, iy, j, k, kk int

	// one = 1.0
	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dspmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || ((alpha == zero) && (beta == one)) {
	// 	return
	// }

	// xiter := x.Iter(n, incx)
	// yiter := y.Iter(n, incy)

	// //     Start the operations. In this version the elements of the array AP
	// //     are accessed sequentially with one pass through AP.
	// //
	// //     First form  y := beta*y.
	// if beta != one {
	// 	if beta == zero {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, zero)
	// 		}
	// 	} else {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, beta*y.Get(i))
	// 		}
	// 	}
	// }
	// if alpha == zero {
	// 	return
	// }
	// kk = 1
	// if uplo == mat.Upper {
	// 	//        Form  y  when AP contains the upper triangle.
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		for k, ix, iy = kk-1, xiter[0], yiter[0]; k < kk+j-1; k, ix, iy = k+1, ix+incx, iy+incy {
	// 			y.Set(iy, y.Get(iy)+temp1*ap.Get(k))
	// 			temp2 += ap.Get(k) * x.Get(ix)
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*ap.Get(kk+j-1)+alpha*temp2)
	// 		kk += j + 1
	// 	}
	// } else {
	// 	//        Form  y  when AP contains the lower triangle.
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*ap.Get(kk-1))
	// 		for k, ix, iy = kk, xiter[j]+incx, yiter[j]+incy; k < kk+n-(j+1); k, ix, iy = k+1, ix+incx, iy+incy {
	// 			y.Set(iy, y.Get(iy)+temp1*ap.Get(k))
	// 			temp2 += ap.Get(k) * x.Get(ix)
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
	// 		kk += n - (j + 1) + 1
	// 	}
	// }

	return y.Spmv(uplo, n, alpha, ap, x, incx, beta, incy)
}

// Dsymv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func Dsymv(uplo mat.MatUplo, n int, alpha float64, a *mat.Matrix, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	// var one, temp1, temp2, zero float64
	// var i, j int

	// one = 1.0
	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, n) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dsymv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || ((alpha == zero) && (beta == one)) {
	// 	return
	// }

	// xiter := x.Iter(n, incx)
	// yiter := y.Iter(n, incy)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through the triangular part
	// //     of A.
	// //
	// //     First form  y := beta*y.
	// if beta != one {
	// 	if beta == zero {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, zero)
	// 		}
	// 	} else {
	// 		for _, i = range y.Iter(n, incy) {
	// 			y.Set(i, beta*y.Get(i))
	// 		}
	// 	}
	// }
	// if alpha == zero {
	// 	return
	// }
	// if uplo == mat.Upper {
	// 	//        Form  y  when A is stored in upper triangle.
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		for i = 0; i < j; i++ {
	// 			y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(i, j))
	// 			temp2 += a.Get(i, j) * x.Get(xiter[i])
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*a.Get(j, j)+alpha*temp2)
	// 	}
	// } else {
	// 	//        Form  y  when A is stored in lower triangle.
	// 	for j = 0; j < n; j++ {
	// 		temp1 = alpha * x.Get(xiter[j])
	// 		temp2 = zero
	// 		y.Set(yiter[j], y.Get(yiter[j])+temp1*a.Get(j, j))
	// 		for i = j + 1; i < n; i++ {
	// 			y.Set(yiter[i], y.Get(yiter[i])+temp1*a.Get(i, j))
	// 			temp2 += a.Get(i, j) * x.Get(xiter[i])
	// 		}
	// 		y.Set(yiter[j], y.Get(yiter[j])+alpha*temp2)
	// 	}
	// }

	return y.Symv(uplo, n, alpha, a, x, incx, beta, incy)
}

// Dtbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func Dtbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.Matrix, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j, kplus1, l int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if k < 0 {
	// 	err = fmt.Errorf("k invalid: %v", k)
	// } else if a.Rows < (k + 1) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtbmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX   too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through A.
	// if trans == mat.NoTrans {
	// 	//         Form  x := A*x.
	// 	if uplo == mat.Upper {
	// 		kplus1 = k + 1
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				l = kplus1 - (j + 1)
	// 				for i = max(0, j-k); i < j; i++ {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(l+i, j))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*a.Get(kplus1-1, j))
	// 				}
	// 			}
	// 		}
	// 	} else {
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				l = 1 - (j + 1)
	// 				for i = min(n-1, j+k); i > j; i-- {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(l+i, j))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*a.Get(0, j))
	// 				}
	// 			}
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := A**T*x.
	// 	if uplo == mat.Upper {
	// 		kplus1 = k + 1
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			l = kplus1 - (j + 1)
	// 			if nounit {
	// 				temp *= a.Get(kplus1-1, j)
	// 			}
	// 			for i = j - 1; i >= max(0, j-k); i-- {
	// 				temp += a.Get(l+i, j) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	} else {
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			l = 1 - (j + 1)
	// 			if nounit {
	// 				temp *= a.Get(0, j)
	// 			}
	// 			for i = j + 1; i < min(n, j+k+1); i++ {
	// 				temp += a.Get(l+i, j) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	}
	// }

	return x.Tbmv(uplo, trans, diag, n, k, a, incx)
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
func Dtbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.Matrix, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j, kplus1, l int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if k < 0 {
	// 	err = fmt.Errorf("k invalid: %v", k)
	// } else if a.Rows < (k + 1) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtbsv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX  too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of A are
	// //     accessed by sequentially with one pass through A.
	// if trans == mat.NoTrans {
	// 	//        Form  x := inv( A )*x.
	// 	if uplo == mat.Upper {
	// 		kplus1 = k
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				l = kplus1 - j
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/a.Get(kplus1, j))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i = j - 1; i >= max(0, j-k); i-- {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(l+i, j))
	// 				}
	// 			}
	// 		}
	// 	} else {
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				l = -j
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/a.Get(0, j))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i = j + 1; i < min(n, j+k+1); i++ {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(l+i, j))
	// 				}
	// 			}
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := inv( A**T)*x.
	// 	if uplo == mat.Upper {
	// 		kplus1 = k
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			l = kplus1 - j
	// 			for i = max(0, j-k); i < j; i++ {
	// 				temp -= a.Get(l+i, j) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= a.Get(kplus1, j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	} else {
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			l = 1 - j - 1
	// 			for i = min(n-1, j+k); i > j; i-- {
	// 				temp -= a.Get(l+i, j) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= a.Get(0, j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	}
	// }

	return x.Tbsv(uplo, trans, diag, n, k, a, incx)
}

// Dtpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func Dtpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j, k, kk int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtpmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX  too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of AP are
	// //     accessed sequentially with one pass through AP.
	// if trans == mat.NoTrans {
	// 	//        Form  x:= A*x.
	// 	if uplo == mat.Upper {
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*ap.Get(k))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*ap.Get(kk+j))
	// 				}
	// 			}
	// 			kk += j + 1
	// 		}
	// 	} else {
	// 		kk = (n * (n + 1)) / 2
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				for i, k = n-1, kk-1; k > kk-n+j; i, k = i-1, k-1 {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*ap.Get(k))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*ap.Get(kk-n+j))
	// 				}
	// 			}
	// 			kk -= (n - j)
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := A**T*x.
	// 	if uplo == mat.Upper {
	// 		kk = (n * (n + 1)) / 2
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			if nounit {
	// 				temp *= ap.Get(kk - 1)
	// 			}
	// 			for i, k = j-1, kk-2; k >= kk-j-1; i, k = i-1, k-1 {
	// 				temp += ap.Get(k) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 			kk -= (j + 1)
	// 		}
	// 	} else {
	// 		kk = 1
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			if nounit {
	// 				temp *= ap.Get(kk - 1)
	// 			}
	// 			for i, k = j+1, kk; k < kk+n-(j+1); i, k = i+1, k+1 {
	// 				temp += ap.Get(k) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 			kk += (n - j)
	// 		}
	// 	}
	// }

	return x.Tpmv(uplo, trans, diag, n, ap, incx)
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
func Dtpsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j, k, kk int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtpsv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX  too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of AP are
	// //     accessed sequentially with one pass through AP.
	// if trans == mat.NoTrans {
	// 	//        Form  x := inv( A )*x.
	// 	if uplo == mat.Upper {
	// 		kk = (n*(n+1))/2 - 1
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/ap.Get(kk))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*ap.Get(k))
	// 				}
	// 			}
	// 			kk -= (j + 1)
	// 		}
	// 	} else {
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/ap.Get(kk))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i, k = j+1, kk+1; k <= kk+n-j-1; i, k = i+1, k+1 {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*ap.Get(k))
	// 				}
	// 			}
	// 			kk += (n - j)
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := inv( A**T )*x.
	// 	if uplo == mat.Upper {
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
	// 				temp -= ap.Get(k) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= ap.Get(kk + j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 			kk += j + 1
	// 		}
	// 	} else {
	// 		kk = (n * (n + 1)) / 2
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
	// 				temp -= ap.Get(k) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= ap.Get(kk - n + j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 			kk -= (n - j)
	// 		}
	// 	}
	// }

	return x.Tpsv(uplo, trans, diag, n, ap, incx)
}

// Dtrmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func Dtrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.Matrix, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j int

	// zero = 0.0
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, n) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtrmv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX  too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through A.
	// if trans == mat.NoTrans {
	// 	//        Form  x := A*x.
	// 	if uplo == mat.Upper {
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				for i = 0; i < j; i++ {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(i, j))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*a.Get(j, j))
	// 				}
	// 			}
	// 		}
	// 	} else {
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				temp = x.Get(xiter[j])
	// 				for i = n - 1; i >= j+1; i-- {
	// 					x.Set(xiter[i], x.Get(xiter[i])+temp*a.Get(i, j))
	// 				}
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])*a.Get(j, j))
	// 				}
	// 			}
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := A**T*x.
	// 	if uplo == mat.Upper {
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			if nounit {
	// 				temp *= a.Get(j, j)
	// 			}
	// 			for i = j - 1; i >= 0; i-- {
	// 				temp += a.Get(i, j) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	} else {
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			if nounit {
	// 				temp *= a.Get(j, j)
	// 			}
	// 			for i = j + 1; i < n; i++ {
	// 				temp += a.Get(i, j) * x.Get(xiter[i])
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	}
	// }

	return x.Trmv(uplo, trans, diag, n, a, incx)
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
func Dtrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.Matrix, x *mat.Vector, incx int) (err error) {
	// var nounit bool
	// var temp, zero float64
	// var i, j int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if !trans.IsValid() {
	// 	err = fmt.Errorf("trans invalid: %v", trans.String())
	// } else if !diag.IsValid() {
	// 	err = fmt.Errorf("diag invalid: %v", diag.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, n) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dtrsv"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if n == 0 {
	// 	return
	// }

	// nounit = diag == mat.NonUnit

	// //     Set up the start point in X if the increment is not unity. This
	// //     will be  ( N - 1 )*INCX  too small for descending loops.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through A.
	// if trans == mat.NoTrans {
	// 	//        Form  x := inv( A )*x.
	// 	if uplo == mat.Upper {
	// 		for j = n - 1; j >= 0; j-- {
	// 			if x.Get(xiter[j]) != zero {
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/a.Get(j, j))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i = j - 1; i >= 0; i-- {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(i, j))
	// 				}
	// 			}
	// 		}
	// 	} else {
	// 		for j = 0; j < n; j++ {
	// 			if x.Get(xiter[j]) != zero {
	// 				if nounit {
	// 					x.Set(xiter[j], x.Get(xiter[j])/a.Get(j, j))
	// 				}
	// 				temp = x.Get(xiter[j])
	// 				for i = j + 1; i < n; i++ {
	// 					x.Set(xiter[i], x.Get(xiter[i])-temp*a.Get(i, j))
	// 				}
	// 			}
	// 		}
	// 	}
	// } else {
	// 	//        Form  x := inv( A**T )*x.
	// 	if uplo == mat.Upper {
	// 		for j = 0; j < n; j++ {
	// 			temp = x.Get(xiter[j])
	// 			for i = 0; i < j; i++ {
	// 				temp -= a.Get(i, j) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= a.Get(j, j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	} else {
	// 		for j = n - 1; j >= 0; j-- {
	// 			temp = x.Get(xiter[j])
	// 			for i = n - 1; i >= j+1; i-- {
	// 				temp -= a.Get(i, j) * x.Get(xiter[i])
	// 			}
	// 			if nounit {
	// 				temp /= a.Get(j, j)
	// 			}
	// 			x.Set(xiter[j], temp)
	// 		}
	// 	}
	// }

	return x.Trsv(uplo, trans, diag, n, a, incx)
}

// Dger performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Dger(m, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, a *mat.Matrix) (err error) {
	// var temp, zero float64
	// var i, j int

	// zero = 0.0
	// //     Test the input parameters.
	// if m < 0 {
	// 	err = fmt.Errorf("m invalid: %v", m)
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, m) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dger"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (m == 0) || (n == 0) || (alpha == zero) {
	// 	return
	// }

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through A.
	// xiter := x.Iter(m, incx)
	// yiter := y.Iter(n, incy)
	// for j = 0; j < n; j++ {
	// 	if y.Get(yiter[j]) != zero {
	// 		temp = alpha * y.Get(yiter[j])
	// 		for i = 0; i < m; i++ {
	// 			a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
	// 		}
	// 	}
	// }

	return a.Ger(m, n, alpha, x, incx, y, incy)
}

// Dspr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, ap *mat.Vector) (err error) {
	// var temp, zero float64
	// var i, j, k, kk int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dspr"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || (alpha == zero) {
	// 	return
	// }

	// //     Set the start point in X if the increment is not unity.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of the array AP
	// //     are accessed sequentially with one pass through AP.
	// if uplo == mat.Upper {
	// 	for j = 0; j < n; j++ {
	// 		if x.Get(xiter[j]) != zero {
	// 			temp = alpha * x.Get(xiter[j])
	// 			for i, k = 0, kk; k <= kk+j; i, k = i+1, k+1 {
	// 				ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp)
	// 			}
	// 		}
	// 		kk += j + 1
	// 	}
	// } else {
	// 	for j = 0; j < n; j++ {
	// 		if x.Get(xiter[j]) != zero {
	// 			temp = alpha * x.Get(xiter[j])
	// 			for i, k = j, kk; k < kk+n-j; i, k = i+1, k+1 {
	// 				ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp)
	// 			}
	// 		}
	// 		kk += n - j
	// 	}
	// }

	return ap.Spr(uplo, n, alpha, x, incx)
}

// Dsyr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func Dsyr(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, a *mat.Matrix) (err error) {
	// var temp, zero float64
	// var i, j int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, n) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dsyr"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || (alpha == zero) {
	// 	return
	// }

	// //     Set the start point in X if the increment is not unity.
	// xiter := x.Iter(n, incx)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through the triangular part
	// //     of A.
	// if uplo == mat.Upper {
	// 	for j = 0; j < n; j++ {
	// 		if x.Get(xiter[j]) != zero {
	// 			temp = alpha * x.Get(xiter[j])
	// 			for i = 0; i < j+1; i++ {
	// 				a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
	// 			}
	// 		}
	// 	}
	// } else {
	// 	for j = 0; j < n; j++ {
	// 		if x.Get(xiter[j]) != zero {
	// 			temp = alpha * x.Get(xiter[j])
	// 			for i = j; i < n; i++ {
	// 				a.Set(i, j, a.Get(i, j)+x.Get(xiter[i])*temp)
	// 			}
	// 		}
	// 	}
	// }

	return a.Syr(uplo, n, alpha, x, incx)
}

// Dspr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr2(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, ap *mat.Vector) (err error) {
	// var temp1, temp2, zero float64
	// var i, j, k, kk int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dspr2"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || (alpha == zero) {
	// 	return
	// }

	// //     Set up the start points in X and Y if the increments are not both
	// //     unity.
	// xiter := x.Iter(n, incx)
	// yiter := y.Iter(n, incy)

	// //     Start the operations. In this version the elements of the array AP
	// //     are accessed sequentially with one pass through AP.
	// kk = 1
	// if uplo == mat.Upper {
	// 	//        Form  A  when upper triangle is stored in AP.
	// 	for j = 0; j < n; j++ {
	// 		if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
	// 			temp1 = alpha * y.Get(yiter[j])
	// 			temp2 = alpha * x.Get(xiter[j])
	// 			for i, k = 0, kk-1; k < kk+j; i, k = i+1, k+1 {
	// 				ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
	// 			}
	// 		}
	// 		kk += j + 1
	// 	}
	// } else {
	// 	//        Form  A  when lower triangle is stored in AP.
	// 	for j = 0; j < n; j++ {
	// 		if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
	// 			temp1 = alpha * y.Get(yiter[j])
	// 			temp2 = alpha * x.Get(xiter[j])
	// 			for i, k = j, kk-1; k < kk+n-j-1; i, k = i+1, k+1 {
	// 				ap.Set(k, ap.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
	// 			}
	// 		}
	// 		kk += n - j
	// 	}
	// }

	return ap.Spr2(uplo, n, alpha, x, incx, y, incy)
}

// Dsyr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n symmetric matrix.
func Dsyr2(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, a *mat.Matrix) (err error) {
	// var temp1, temp2, zero float64
	// var i, j int

	// zero = 0.0

	// //     Test the input parameters.
	// if !uplo.IsValid() {
	// 	err = fmt.Errorf("uplo invalid: %v", uplo.String())
	// } else if n < 0 {
	// 	err = fmt.Errorf("n invalid: %v", n)
	// } else if a.Rows < max(1, n) {
	// 	err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	// }
	// if err != nil {
	// 	Xerbla2([]byte("Dsyr2"), err)
	// 	return
	// }

	// //     Quick return if possible.
	// if (n == 0) || (alpha == zero) {
	// 	return
	// }

	// //     Set up the start points in X and Y if the increments are not both
	// //     unity.
	// xiter := x.Iter(n, incx)
	// yiter := y.Iter(n, incy)

	// //     Start the operations. In this version the elements of A are
	// //     accessed sequentially with one pass through the triangular part
	// //     of A.
	// if uplo == Upper {
	// 	//        Form  A  when A is stored in the upper triangle.
	// 	for j = 1; j <= n; j++ {
	// 		if (x.Get(xiter[j-1]) != zero) || (y.Get(yiter[j-1]) != zero) {
	// 			temp1 = alpha * y.Get(yiter[j-1])
	// 			temp2 = alpha * x.Get(xiter[j-1])
	// 			for i = 1; i <= j; i++ {
	// 				a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(xiter[i-1])*temp1+y.Get(yiter[i-1])*temp2)
	// 			}
	// 		}
	// 	}
	// } else {
	// 	//        Form  A  when A is stored in the lower triangle.
	// 	for j = 1; j <= n; j++ {
	// 		if (x.Get(xiter[j-1]) != zero) || (y.Get(yiter[j-1]) != zero) {
	// 			temp1 = alpha * y.Get(yiter[j-1])
	// 			temp2 = alpha * x.Get(xiter[j-1])
	// 			for i = j; i <= n; i++ {
	// 				a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(xiter[i-1])*temp1+y.Get(yiter[i-1])*temp2)
	// 			}
	// 		}
	// 	}
	// }

	return a.Syr2(uplo, n, alpha, x, incx, y, incy)
}
