package goblas

import (
	"fmt"

	"github.com/whipstein/golinalg/mat"
)

// Dgemm performs one of the matrix-matrix operations
//    C := alpha*op( A )*op( B ) + beta*C,
// where  op( X ) is one of
//    op( X ) = X   or   op( X ) = X**T,
// alpha and beta are scalars, and A, B and C are matrices, with op( A )
// an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
func Dgemm(transa, transb mat.MatTrans, m, n, k int, alpha float64, a *mat.Matrix, b *mat.Matrix, beta float64, c *mat.Matrix) (err error) {
	var nota, notb bool
	var one, temp, zero float64
	var i, j, l, nrowa, nrowb int

	one = 1.0
	zero = 0.0

	//     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
	//     transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
	//     and  columns of  A  and the  number of  rows  of  B  respectively.
	nota = transa == mat.NoTrans
	notb = transb == mat.NoTrans
	if nota {
		nrowa = m
		// ncola = k
	} else {
		nrowa = k
		// ncola = m
	}
	if notb {
		nrowb = k
	} else {
		nrowb = n
	}

	//     Test the input parameters.
	if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !transb.IsValid() {
		err = fmt.Errorf("transb invalid: %v", transb.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowb) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowb))
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", c.Rows, max(1, m))
	}
	if err != nil {
		Xerbla2([]byte("Dgemm"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And if  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if notb {
		if nota {
			//           Form  C := alpha*A*B + beta*C.
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(l-1, j-1)
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
					}
				}
			}
		}
	} else {
		if nota {
			//           Form  C := alpha*A*B**T + beta*C
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(j-1, l-1)
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B**T + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(j-1, l-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Dsymm performs one of the matrix-matrix operations
//    C := alpha*A*B + beta*C,
// or
//    C := alpha*B*A + beta*C,
// where alpha and beta are scalars,  A is a symmetric matrix and  B and
// C are  m by n matrices.
func Dsymm(side mat.MatSide, uplo mat.MatUplo, m, n int, alpha float64, a *mat.Matrix, b *mat.Matrix, beta float64, c *mat.Matrix) (err error) {
	var upper bool
	var one, temp1, temp2, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Set NROWA as the number of rows of A.
	if side == mat.Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == mat.Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", c.Rows, max(1, m))
	}
	if err != nil {
		Xerbla2([]byte("Dsymm"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}
	//
	//     And when  alpha.eq.zero.
	//
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if side == mat.Left {
		//        Form  C := alpha*A*B + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = 1; k <= i-1; k++ {
						c.Set(k-1, j-1, c.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = m; i >= 1; i-- {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = i + 1; k <= m; k++ {
						c.Set(k-1, j-1, c.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*B*A + beta*C.
		for j = 1; j <= n; j++ {
			temp1 = alpha * a.Get(j-1, j-1)
			if beta == zero {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, temp1*b.Get(i-1, j-1))
				}
			} else {
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+temp1*b.Get(i-1, j-1))
				}
			}
			for k = 1; k <= j-1; k++ {
				if upper {
					temp1 = alpha * a.Get(k-1, j-1)
				} else {
					temp1 = alpha * a.Get(j-1, k-1)
				}
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, c.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
			for k = j + 1; k <= n; k++ {
				if upper {
					temp1 = alpha * a.Get(j-1, k-1)
				} else {
					temp1 = alpha * a.Get(k-1, j-1)
				}
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, c.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
		}
	}

	return
}

// Dtrmm performs one of the matrix-matrix operations
//    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
// where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//    op( A ) = A   or   op( A ) = A**T.
func Dtrmm(side mat.MatSide, uplo mat.MatUplo, transa mat.MatTrans, diag mat.MatDiag, m, n int, alpha float64, a *mat.Matrix, b *mat.Matrix) (err error) {
	var lside, nounit, upper bool
	var one, temp, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	lside = side == mat.Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	nounit = diag == mat.NonUnit
	upper = uplo == mat.Upper

	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	}
	if err != nil {
		Xerbla2([]byte("Dtrmm"), err)
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}
	//
	//     And when  alpha.eq.zero.
	//
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == mat.NoTrans {
			//           Form  B := alpha*A*B.
			if upper {
				for j = 1; j <= n; j++ {
					for k = 1; k <= m; k++ {
						if b.Get(k-1, j-1) != zero {
							temp = alpha * b.Get(k-1, j-1)
							for i = 1; i <= k-1; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
							if nounit {
								temp *= a.Get(k-1, k-1)
							}
							b.Set(k-1, j-1, temp)
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for k = m; k >= 1; k-- {
						if b.Get(k-1, j-1) != zero {
							temp = alpha * b.Get(k-1, j-1)
							b.Set(k-1, j-1, temp)
							if nounit {
								b.Set(k-1, j-1, b.Get(k-1, j-1)*a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*A**T*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = b.Get(i-1, j-1)
						if nounit {
							temp *= a.Get(i-1, i-1)
						}
						for k = 1; k <= i-1; k++ {
							temp += a.Get(k-1, i-1) * b.Get(k-1, j-1)
						}
						b.Set(i-1, j-1, alpha*temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = b.Get(i-1, j-1)
						if nounit {
							temp *= a.Get(i-1, i-1)
						}
						for k = i + 1; k <= m; k++ {
							temp += a.Get(k-1, i-1) * b.Get(k-1, j-1)
						}
						b.Set(i-1, j-1, alpha*temp)
					}
				}
			}
		}
	} else {
		if transa == mat.NoTrans {
			//           Form  B := alpha*B*A.
			if upper {
				for j = n; j >= 1; j-- {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						b.Set(i-1, j-1, temp*b.Get(i-1, j-1))
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						b.Set(i-1, j-1, temp*b.Get(i-1, j-1))
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*A**T.
			if upper {
				for k = 1; k <= n; k++ {
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = alpha * a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						temp *= a.Get(k-1, k-1)
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = n; k >= 1; k-- {
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = alpha * a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						temp *= a.Get(k-1, k-1)
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Dtrsm solves one of the matrix equations
//    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//    op( A ) = A   or   op( A ) = A**T.
// The matrix X is overwritten on B.
func Dtrsm(side mat.MatSide, uplo mat.MatUplo, transa mat.MatTrans, diag mat.MatDiag, m, n int, alpha float64, a *mat.Matrix, b *mat.Matrix) (err error) {
	var lside, nounit, upper bool
	var one, temp, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	lside = side == mat.Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	nounit = diag == mat.NonUnit
	upper = uplo == mat.Upper

	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	}
	if err != nil {
		Xerbla2([]byte("Dtrsm"), err)
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == mat.NoTrans {
			//           Form  B := alpha*inv( A )*B.
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, alpha*b.Get(i-1, j-1))
						}
					}
					for k = m; k >= 1; k-- {
						if b.Get(k-1, j-1) != zero {
							if nounit {
								b.Set(k-1, j-1, b.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = 1; i <= k-1; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, alpha*b.Get(i-1, j-1))
						}
					}
					for k = 1; k <= m; k++ {
						if b.Get(k-1, j-1) != zero {
							if nounit {
								b.Set(k-1, j-1, b.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-b.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*inv( A**T )*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = alpha * b.Get(i-1, j-1)
						for k = 1; k <= i-1; k++ {
							temp -= a.Get(k-1, i-1) * b.Get(k-1, j-1)
						}
						if nounit {
							temp /= a.Get(i-1, i-1)
						}
						b.Set(i-1, j-1, temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = alpha * b.Get(i-1, j-1)
						for k = i + 1; k <= m; k++ {
							temp -= a.Get(k-1, i-1) * b.Get(k-1, j-1)
						}
						if nounit {
							temp /= a.Get(i-1, i-1)
						}
						b.Set(i-1, j-1, temp)
					}
				}
			}
		}
	} else {
		if transa == mat.NoTrans {
			//           Form  B := alpha*B*inv( A ).
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, alpha*b.Get(i-1, j-1))
						}
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-a.Get(k-1, j-1)*b.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, temp*b.Get(i-1, j-1))
						}
					}
				}
			} else {
				for j = n; j >= 1; j-- {
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, alpha*b.Get(i-1, j-1))
						}
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-a.Get(k-1, j-1)*b.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							b.Set(i-1, j-1, temp*b.Get(i-1, j-1))
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*inv( A**T ).
			if upper {
				for k = n; k >= 1; k-- {
					if nounit {
						temp = one / a.Get(k-1, k-1)
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-temp*b.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, alpha*b.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = 1; k <= n; k++ {
					if nounit {
						temp = one / a.Get(k-1, k-1)
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)-temp*b.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, alpha*b.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Dsyrk performs one of the symmetric rank k operations
//    C := alpha*A*A**T + beta*C,
// or
//    C := alpha*A**T*A + beta*C,
// where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
// and  A  is an  n by k  matrix in the first case and a  k by n  matrix
// in the second case.
func Dsyrk(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha float64, a *mat.Matrix, beta float64, c *mat.Matrix) (err error) {
	var upper bool
	var one, temp, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == mat.NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == mat.Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if c.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", c.Rows, max(1, n))
	}
	if err != nil {
		Xerbla2([]byte("Dsyrk"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == mat.NoTrans {
		//        Form  C := alpha*A*A**T + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*A + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Dsyr2k performs one of the symmetric rank 2k operations
//    C := alpha*A*B**T + alpha*B*A**T + beta*C,
// or
//    C := alpha*A**T*B + alpha*B**T*A + beta*C,
// where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
// and  A and B  are  n by k  matrices  in the  first  case  and  k by n
// matrices in the second case.
func Dsyr2k(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha float64, a *mat.Matrix, b *mat.Matrix, beta float64, c *mat.Matrix) (err error) {
	var upper bool
	var one, temp1, temp2, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == mat.NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == mat.Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowa) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowa))
	} else if c.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", c.Rows, max(1, n))
	}
	if err != nil {
		Xerbla2([]byte("Dsyr2k"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == mat.NoTrans {
		//        Form  C := alpha*A*B**T + alpha*B*A**T + C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*B + alpha*B**T*A + C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 += a.Get(l-1, i-1) * b.Get(l-1, j-1)
						temp2 += b.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 += a.Get(l-1, i-1) * b.Get(l-1, j-1)
						temp2 += b.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		}
	}

	return
}
