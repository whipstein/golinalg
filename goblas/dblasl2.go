package goblas

import (
	"fmt"

	"github.com/whipstein/golinalg/mat"
)

// Dgbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func Dgbmv(trans mat.MatTrans, m, n, kl, ku int, alpha float64, a *mat.Matrix, lda int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	var one, temp, zero float64
	var i, ix, iy, j, jx, jy, k, kup1, kx, ky, lenx, leny int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
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
	if err != nil {
		Xerbla2([]byte("Dgbmv"), err)
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 || (alpha == zero && beta == one) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == mat.NoTrans {
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
	if trans == mat.NoTrans {
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
		//        Form  y := alpha*A**T*x + y.
		for j, jy = 1, ky; j <= n; j, jy = j+1, jy+incy {
			temp = zero
			k = kup1 - j
			for i, ix = max(1, j-ku), kx; i <= min(m, j+kl); i, ix = i+1, ix+incx {
				temp += a.Get(k+i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp)
			if j > ku {
				kx += incx
			}
		}
	}

	return
}

// Dgemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func Dgemv(trans mat.MatTrans, m, n int, alpha float64, a *mat.Matrix, lda int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	var one, temp, zero float64
	var i, ix, iy, j, jx, jy, kx, ky, lenx, leny int

	one = 1.0
	zero = 0.0

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
		Xerbla2([]byte("Dgemv"), err)
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 || (alpha == zero && beta == one) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == mat.NoTrans {
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
	if trans == mat.NoTrans {
		//        Form  y := alpha*A*x + y.
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			temp = alpha * x.Get(jx-1)
			for i, iy = 1, ky; i <= m; i, iy = i+1, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp*a.Get(i-1, j-1))
			}
		}
	} else {
		for j, jy = 1, ky; j <= n; j, jy = j+1, jy+incy {
			temp = zero
			for i, ix = 1, kx; i <= m; i, ix = i+1, ix+incx {
				temp += a.Get(i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp)
		}
	}

	// blocksize := 512

	// if n < minParBlocks*blocksize {
	// 	dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
	// } else {
	// 	nblocks := blocks(n, blocksize)
	// 	var wg sync.WaitGroup
	// 	defer wg.Wait()

	// 	for i := 0; i < nblocks; i++ {
	// 		size := blocksize
	// 		if (i+1)*blocksize > n {
	// 			size = n - i*blocksize
	// 		}
	// 		wg.Add(1)
	// 		go func(i, size int) {
	// 			defer wg.Done()
	// 			dgemv(trans, m, n, alpha, a.Off(), lda, x.Off(i*blocksize*incx), incx, beta, y.Off(i*blocksize*incy), incy)
	// 		}(i, size)
	// 	}
	// }

	return
}

// func dgemv(trans mat.MatTrans, m, n int, alpha float64, a *mat.Matrix, lda int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
// 	var one, temp, zero float64
// 	var i, ix, iy, j, jx, jy, kx, ky, leny int

// 	one = 1.0
// 	zero = 0.0

// 	//     Start the operations. In this version the elements of A are
// 	//     accessed sequentially with one pass through A.
// 	//
// 	//     First form  y := beta*y.
// 	if beta != one {
// 		iy = ky
// 		if beta == zero {
// 			for i = 1; i <= leny; i, iy = i+1, iy+incy {
// 				y.Set(iy-1, zero)
// 			}
// 		} else {
// 			for i = 1; i <= leny; i, iy = i+1, iy+incy {
// 				y.Set(iy-1, beta*y.Get(iy-1))
// 			}
// 		}
// 	}
// 	if alpha == zero {
// 		return
// 	}
// 	if trans == mat.NoTrans {
// 		//        Form  y := alpha*A*x + y.
// 		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
// 			temp = alpha * x.Get(jx-1)
// 			for i, iy = 1, ky; i <= m; i, iy = i+1, iy+incy {
// 				y.Set(iy-1, y.Get(iy-1)+temp*a.Get(i-1, j-1))
// 			}
// 		}
// 	} else {
// 		for j, jy = 1, ky; j <= n; j, jy = j+1, jy+incy {
// 			temp = zero
// 			for i, ix = 1, kx; i <= m; i, ix = i+1, ix+incx {
// 				temp += a.Get(i-1, j-1) * x.Get(ix-1)
// 			}
// 			y.Set(jy-1, y.Get(jy-1)+alpha*temp)
// 		}
// 	}

// 	return
// }

// Dsbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric band matrix, with k super-diagonals.
func Dsbmv(uplo mat.MatUplo, n, k int, alpha float64, a *mat.Matrix, lda int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	var one, temp1, temp2, zero float64
	var i, ix, iy, j, jx, jy, kplus1, kx, ky, l int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
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
		Xerbla2([]byte("Dsbmv"), err)
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
	if uplo == mat.Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = k + 1
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			l = kplus1 - j
			for i, ix, iy = max(1, j-k), kx, ky; i <= j-1; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
				temp2 += a.Get(l+i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(kplus1-1, j-1)+alpha*temp2)
			if j > k {
				kx += incx
				ky += incy
			}
		}
	} else {
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(0, j-1))
			l = 1 - j
			for i, ix, iy = j+1, jx+incx, jy+incy; i <= min(n, j+k); i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(l+i-1, j-1))
				temp2 += a.Get(l+i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
		}
	}

	return
}

// Dspmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix, supplied in packed form.
func Dspmv(uplo mat.MatUplo, n int, alpha float64, ap, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	var one, temp1, temp2, zero float64
	var i, ix, iy, j, jx, jy, k, kk, kx, ky int

	one = 1.0
	zero = 0.0

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
		Xerbla2([]byte("Dspmv"), err)
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
	if uplo == mat.Upper {
		//        Form  y  when AP contains the upper triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			for k, ix, iy = kk, kx, ky; k <= kk+j-2; k, ix, iy = k+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
				temp2 += ap.Get(k-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*ap.Get(kk+j-1-1)+alpha*temp2)
			kk += j
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*ap.Get(kk-1))
			for k, ix, iy = kk+1, jx+incx, jy+incy; k <= kk+n-j; k, ix, iy = k+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*ap.Get(k-1))
				temp2 += ap.Get(k-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
			kk += n - j + 1
		}
	}

	return
}

// Dsymv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func Dsymv(uplo mat.MatUplo, n int, alpha float64, a *mat.Matrix, lda int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int) (err error) {
	var one, temp1, temp2, zero float64
	var i, ix, iy, j, jx, jy, kx, ky int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
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
		Xerbla2([]byte("Dsymv"), err)
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
	if uplo == mat.Upper {
		//        Form  y  when A is stored in upper triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			for i, ix, iy = 1, kx, ky; i <= j-1; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
				temp2 += a.Get(i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1)+alpha*temp2)
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		for j, jx, jy = 1, kx, ky; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			temp1 = alpha * x.Get(jx-1)
			temp2 = zero
			y.Set(jy-1, y.Get(jy-1)+temp1*a.Get(j-1, j-1))
			for i, ix, iy = j+1, jx+incx, jy+incy; i <= n; i, ix, iy = i+1, ix+incx, iy+incy {
				y.Set(iy-1, y.Get(iy-1)+temp1*a.Get(i-1, j-1))
				temp2 += a.Get(i-1, j-1) * x.Get(ix-1)
			}
			y.Set(jy-1, y.Get(jy-1)+alpha*temp2)
		}
	}

	return
}

// Dtbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func Dtbmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.Matrix, lda int, x *mat.Vector, incx int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, ix, j, jx, kplus1, kx, l int

	zero = 0.0

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
		Xerbla2([]byte("Dtbmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//         Form  x := A*x.
		if uplo == mat.Upper {
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
					kx += incx
				}
			}
		} else {
			kx += (n - 1) * incx
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
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			kplus1 = k + 1
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				kx -= incx
				l = kplus1 - j
				if nounit {
					temp *= a.Get(kplus1-1, j-1)
				}
				for i, ix = j-1, kx; i >= max(1, j-k); i, ix = i-1, ix-incx {
					temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				kx += incx
				l = 1 - j
				if nounit {
					temp *= a.Get(0, j-1)
				}
				for i, ix = j+1, kx; i <= min(n, j+k); i, ix = i+1, ix+incx {
					temp += a.Get(l+i-1, j-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
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
func Dtbsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, k int, a *mat.Matrix, lda int, x *mat.Vector, incx int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, ix, j, jx, kplus1, kx, l int

	zero = 0.0

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
		Xerbla2([]byte("Dtbsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
			kplus1 = k + 1
			kx += (n - 1) * incx
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
		//        Form  x := inv( A**T)*x.
		if uplo == mat.Upper {
			kplus1 = k + 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				l = kplus1 - j
				for i, ix = max(1, j-k), kx; i <= j-1; i, ix = i+1, ix+incx {
					temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= a.Get(kplus1-1, j-1)
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
				l = 1 - j
				for i, ix = min(n, j+k), kx; i >= j+1; i, ix = i-1, ix-incx {
					temp -= a.Get(l+i-1, j-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= a.Get(0, j-1)
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

// Dtpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func Dtpmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, ap, x *mat.Vector, incx int) (err error) {
	var nounit bool
	var temp, zero float64
	var ix, j, jx, k, kk, kx int

	zero = 0.0

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
		Xerbla2([]byte("Dtpmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//        Form  x:= A*x.
		if uplo == mat.Upper {
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
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			kk = (n * (n + 1)) / 2
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				if nounit {
					temp *= ap.Get(kk - 1)
				}
				for k, ix = kk-1, jx-incx; k >= kk-j+1; k, ix = k-1, ix-incx {
					temp += ap.Get(k-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
				kk -= j
			}
		} else {
			kk = 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				if nounit {
					temp *= ap.Get(kk - 1)
				}
				for k, ix = kk+1, jx+incx; k <= kk+n-j; k, ix = k+1, ix+incx {
					temp += ap.Get(k-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
				kk += (n - j + 1)
			}
		}
	}

	return
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
	var nounit bool
	var temp, zero float64
	var ix, j, jx, k, kk, kx int

	zero = 0.0

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
		Xerbla2([]byte("Dtpsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
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
		//        Form  x := inv( A**T )*x.
		if uplo == mat.Upper {
			kk = 1
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				for k, ix = kk, kx; k <= kk+j-2; k, ix = k+1, ix+incx {
					temp -= ap.Get(k-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= ap.Get(kk + j - 1 - 1)
				}
				x.Set(jx-1, temp)
				kk += j
			}
		} else {
			kk = (n * (n + 1)) / 2
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				for k, ix = kk, kx; k >= kk-(n-(j+1)); k, ix = k-1, ix-incx {
					temp -= ap.Get(k-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= ap.Get(kk - n + j - 1)
				}
				x.Set(jx-1, temp)
				kk -= (n - j + 1)
			}
		}
	}

	return
}

// Dtrmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func Dtrmv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.Matrix, lda int, x *mat.Vector, incx int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, ix, j, jx, kx int

	zero = 0.0
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
		Xerbla2([]byte("Dtrmv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//        Form  x := A*x.
		if uplo == mat.Upper {
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
		//        Form  x := A**T*x.
		if uplo == mat.Upper {
			for j, jx = n, kx+(n-1)*incx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				if nounit {
					temp *= a.Get(j-1, j-1)
				}
				for i, ix = j-1, jx-incx; i >= 1; i, ix = i-1, ix-incx {
					temp += a.Get(i-1, j-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
			}
		} else {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				if nounit {
					temp *= a.Get(j-1, j-1)
				}
				for i, ix = j+1, jx+incx; i <= n; i, ix = i+1, ix+incx {
					temp += a.Get(i-1, j-1) * x.Get(ix-1)
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
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
func Dtrsv(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n int, a *mat.Matrix, lda int, x *mat.Vector, incx int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, ix, j, jx, kx int

	zero = 0.0

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
		Xerbla2([]byte("Dtrsv"), err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == mat.NonUnit

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
	if trans == mat.NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == mat.Upper {
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
		//        Form  x := inv( A**T )*x.
		if uplo == mat.Upper {
			for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
				temp = x.Get(jx - 1)
				for i, ix = 1, kx; i <= j-1; i, ix = i+1, ix+incx {
					temp -= a.Get(i-1, j-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= a.Get(j-1, j-1)
				}
				x.Set(jx-1, temp)
			}
		} else {
			kx += (n - 1) * incx
			for j, jx = n, kx; j >= 1; j, jx = j-1, jx-incx {
				temp = x.Get(jx - 1)
				for i, ix = n, kx; i >= j+1; i, ix = i-1, ix-incx {
					temp -= a.Get(i-1, j-1) * x.Get(ix-1)
				}
				if nounit {
					temp /= a.Get(j-1, j-1)
				}
				x.Set(jx-1, temp)
			}
		}
	}

	return
}

// Dger performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func Dger(m, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, a *mat.Matrix, lda int) (err error) {
	var temp, zero float64
	var i, ix, j, jy, kx int

	zero = 0.0
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
		Xerbla2([]byte("Dger"), err)
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

// Dspr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, ap *mat.Vector) (err error) {
	var temp, zero float64
	var ix, j, jx, k, kk, kx int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if incx == 0 {
		err = fmt.Errorf("incx invalid: %v", incx)
	}
	if err != nil {
		Xerbla2([]byte("Dspr"), err)
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
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == mat.Upper {
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = alpha * x.Get(jx-1)
				for k, ix = kk, kx; k <= kk+j-1; k, ix = k+1, ix+incx {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
				}
			}
			kk += j
		}
	} else {
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = alpha * x.Get(jx-1)
				ix = jx
				for k, ix = kk, jx; k <= kk+n-j; k, ix = k+1, ix+incx {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp)
				}
			}
			kk += n - j + 1
		}
	}

	return
}

// Dsyr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func Dsyr(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, a *mat.Matrix, lda int) (err error) {
	var temp, zero float64
	var i, ix, j, jx, kx int

	zero = 0.0

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
		Xerbla2([]byte("Dsyr"), err)
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
	if incx == 1 {
		kx++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == mat.Upper {
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = alpha * x.Get(jx-1)
				for i, ix = 1, kx; i <= j; i, ix = i+1, ix+incx {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
				}
			}
		}
	} else {
		for j, jx = 1, kx; j <= n; j, jx = j+1, jx+incx {
			if x.Get(jx-1) != zero {
				temp = alpha * x.Get(jx-1)
				for i, ix = j, jx; i <= n; i, ix = i+1, ix+incx {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp)
				}
			}
		}
	}

	return
}

// Dspr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n symmetric matrix, supplied in packed form.
func Dspr2(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, ap *mat.Vector) (err error) {
	var temp1, temp2, zero float64
	var ix, iy, j, jx, jy, k, kk, kx, ky int

	zero = 0.0

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
		Xerbla2([]byte("Dspr2"), err)
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
	if uplo == mat.Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.Get(jy-1)
				temp2 = alpha * x.Get(jx-1)
				for k, ix, iy = kk, kx, ky; k <= kk+j-1; k, ix, iy = k+1, ix+incx, iy+incy {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			}
			kk += j
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.Get(jy-1)
				temp2 = alpha * x.Get(jx-1)
				for k, ix, iy = kk, jx, jy; k <= kk+n-j; k, ix, iy = k+1, ix+incx, iy+incy {
					ap.Set(k-1, ap.Get(k-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			}
			kk += n - j + 1
		}
	}

	return
}

// Dsyr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n symmetric matrix.
func Dsyr2(uplo mat.MatUplo, n int, alpha float64, x *mat.Vector, incx int, y *mat.Vector, incy int, a *mat.Matrix, lda int) (err error) {
	var temp1, temp2, zero float64
	var i, ix, iy, j, jx, jy, kx, ky int

	zero = 0.0

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
		Xerbla2([]byte("Dsyr2"), err)
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
		jx++
		jy++
		kx++
		ky++
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == mat.Upper {
		//        Form  A  when A is stored in the upper triangle.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.Get(jy-1)
				temp2 = alpha * x.Get(jx-1)
				for i, ix, iy = 1, kx, ky; i <= j; i, ix, iy = i+1, ix+incx, iy+incy {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		for j = 1; j <= n; j, jx, jy = j+1, jx+incx, jy+incy {
			if (x.Get(jx-1) != zero) || (y.Get(jy-1) != zero) {
				temp1 = alpha * y.Get(jy-1)
				temp2 = alpha * x.Get(jx-1)
				for i, ix, iy = j, jx, jy; i <= n; i, ix, iy = i+1, ix+incx, iy+incy {
					a.Set(i-1, j-1, a.Get(i-1, j-1)+x.Get(ix-1)*temp1+y.Get(iy-1)*temp2)
				}
			}
		}
	}

	return
}
