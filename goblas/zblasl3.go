package goblas

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zgemm performs one of the matrix-matrix operations
//
//    C := alpha*op( A )*op( B ) + beta*C,
//
// where  op( X ) is one of
//
//    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
//
// alpha and beta are scalars, and A, B and C are matrices, with op( A )
// an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
func Zgemm(transa, transb mat.MatTrans, m, n, k int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta complex128, c *mat.CMatrix, ldc int) (err error) {
	var conja, conjb, nota, notb bool
	var one, temp, zero complex128
	var i, j, l, nrowa, nrowb int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
	//     conjugated or transposed, set  CONJA and CONJB  as true if  A  and
	//     B  respectively are to be  transposed but  not conjugated  and set
	//     NROWA, NCOLA and  NROWB  as the number of rows and  columns  of  A
	//     and the number of rows of  B  respectively.
	nota = transa == NoTrans
	notb = transb == NoTrans
	conja = transa == ConjTrans
	conjb = transb == ConjTrans
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
	if (!nota) && (!conja) && (transa != Trans) {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if (!notb) && (!conjb) && (transb != Trans) {
		err = fmt.Errorf("transb invalid: %v", transb.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, nrowb) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	} else if ldc < max(1, m) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zgemm"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
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
		} else if conja {
			//           Form  C := alpha*A**H*B + beta*C.
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * b.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
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
	} else if nota {
		if conjb {
			//           Form  C := alpha*A*B**H + beta*C.
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
					temp = alpha * b.GetConj(j-1, l-1)
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
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
		}
	} else if conja {
		if conjb {
			//           Form  C := alpha*A**H*B**H + beta*C.
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp = temp + a.GetConj(l-1, i-1)*b.GetConj(j-1, l-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**H*B**T + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp = temp + a.GetConj(l-1, i-1)*b.Get(j-1, l-1)
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
		if conjb {
			//           Form  C := alpha*A**T*B**H + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.GetConj(j-1, l-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, alpha*temp)
					} else {
						c.Set(i-1, j-1, alpha*temp+beta*c.Get(i-1, j-1))
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

// Zhemm performs one of the matrix-matrix operations
//
//    C := alpha*A*B + beta*C,
//
// or
//
//    C := alpha*B*A + beta*C,
//
// where alpha and beta are scalars, A is an hermitian matrix and  B and
// C are m by n matrices.
func Zhemm(side mat.MatSide, uplo mat.MatUplo, m, n int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta complex128, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set NROWA as the number of rows of A.
	if side == Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, m) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	} else if ldc < max(1, m) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zhemm"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
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
	if side == Left {
		//        Form  C := alpha*A*B + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = 1; k <= i-1; k++ {
						c.Set(k-1, j-1, c.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.GetConj(k-1, i-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
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
						temp2 += b.Get(k-1, j-1) * a.GetConj(k-1, i-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					} else {
						c.Set(i-1, j-1, beta*c.Get(i-1, j-1)+temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*B*A + beta*C.
		for j = 1; j <= n; j++ {
			temp1 = alpha * a.GetReCmplx(j-1, j-1)
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
					temp1 = alpha * a.GetConj(j-1, k-1)
				}
				for i = 1; i <= m; i++ {
					c.Set(i-1, j-1, c.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
			for k = j + 1; k <= n; k++ {
				if upper {
					temp1 = alpha * a.GetConj(j-1, k-1)
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

// Zsymm performs one of the matrix-matrix operations
//
//    C := alpha*A*B + beta*C,
//
// or
//
//    C := alpha*B*A + beta*C,
//
// where  alpha and beta are scalars, A is a symmetric matrix and  B and
// C are m by n matrices.
func Zsymm(side mat.MatSide, uplo mat.MatUplo, m, n int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta complex128, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set NROWA as the number of rows of A.
	if side == Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, m) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	} else if ldc < max(1, m) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zsymm"), err)
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
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
	if side == Left {
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

// Ztrmm performs one of the matrix-matrix operations
//
//    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
//
// where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//
//    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
func Ztrmm(side mat.MatSide, uplo mat.MatUplo, transa mat.MatTrans, diag mat.MatDiag, m, n int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int) (err error) {
	var lside, noconj, nounit, upper bool
	var one, temp, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	noconj = transa == Trans
	nounit = diag == NonUnit
	upper = uplo == Upper

	if (!lside) && (side != Right) {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if (!upper) && (uplo != Lower) {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if (transa != NoTrans) && (transa != Trans) && (transa != ConjTrans) {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if (diag != Unit) && (diag != NonUnit) {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, m) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	}
	if err != nil {
		Xerbla2([]byte("Ztrmm"), err)
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
		if transa == NoTrans {
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
								temp = temp * a.Get(k-1, k-1)
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
			//           Form  B := alpha*A**T*B   or   B := alpha*A**H*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = b.Get(i-1, j-1)
						if noconj {
							if nounit {
								temp = temp * a.Get(i-1, i-1)
							}
							for k = 1; k <= i-1; k++ {
								temp = temp + a.Get(k-1, i-1)*b.Get(k-1, j-1)
							}
						} else {
							if nounit {
								temp = temp * a.GetConj(i-1, i-1)
							}
							for k = 1; k <= i-1; k++ {
								temp = temp + a.GetConj(k-1, i-1)*b.Get(k-1, j-1)
							}
						}
						b.Set(i-1, j-1, alpha*temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = b.Get(i-1, j-1)
						if noconj {
							if nounit {
								temp = temp * a.Get(i-1, i-1)
							}
							for k = i + 1; k <= m; k++ {
								temp = temp + a.Get(k-1, i-1)*b.Get(k-1, j-1)
							}
						} else {
							if nounit {
								temp = temp * a.GetConj(i-1, i-1)
							}
							for k = i + 1; k <= m; k++ {
								temp = temp + a.GetConj(k-1, i-1)*b.Get(k-1, j-1)
							}
						}
						b.Set(i-1, j-1, alpha*temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
			//           Form  B := alpha*B*A.
			if upper {
				for j = n; j >= 1; j-- {
					temp = alpha
					if nounit {
						temp = temp * a.Get(j-1, j-1)
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
						temp = temp * a.Get(j-1, j-1)
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
			//           Form  B := alpha*B*A**T   or   B := alpha*B*A**H.
			if upper {
				for k = 1; k <= n; k++ {
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = alpha * a.Get(j-1, k-1)
							} else {
								temp = alpha * a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						if noconj {
							temp = temp * a.Get(k-1, k-1)
						} else {
							temp = temp * a.GetConj(k-1, k-1)
						}
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
							if noconj {
								temp = alpha * a.Get(j-1, k-1)
							} else {
								temp = alpha * a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								b.Set(i-1, j-1, b.Get(i-1, j-1)+temp*b.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						if noconj {
							temp = temp * a.Get(k-1, k-1)
						} else {
							temp = temp * a.GetConj(k-1, k-1)
						}
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

// goblas.Ztrsm solves one of the matrix equations
//
//    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
//
// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//
//    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
//
// The matrix X is overwritten on B.
func Ztrsm(side mat.MatSide, uplo mat.MatUplo, transa mat.MatTrans, diag mat.MatDiag, m, n int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int) (err error) {
	var lside, noconj, nounit, upper bool
	var one, temp, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	noconj = transa == Trans
	nounit = diag == NonUnit
	upper = uplo == Upper

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
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, m) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	}
	if err != nil {
		Xerbla2([]byte("Ztrsm"), err)
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
		if transa == NoTrans {
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
			//           Form  B := alpha*inv( A**T )*B
			//           or    B := alpha*inv( A**H )*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = alpha * b.Get(i-1, j-1)
						if noconj {
							for k = 1; k <= i-1; k++ {
								temp = temp - a.Get(k-1, i-1)*b.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.Get(i-1, i-1)
							}
						} else {
							for k = 1; k <= i-1; k++ {
								temp = temp - a.GetConj(k-1, i-1)*b.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.GetConj(i-1, i-1)
							}
						}
						b.Set(i-1, j-1, temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = alpha * b.Get(i-1, j-1)
						if noconj {
							for k = i + 1; k <= m; k++ {
								temp = temp - a.Get(k-1, i-1)*b.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.Get(i-1, i-1)
							}
						} else {
							for k = i + 1; k <= m; k++ {
								temp = temp - a.GetConj(k-1, i-1)*b.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.GetConj(i-1, i-1)
							}
						}
						b.Set(i-1, j-1, temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
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
			//           Form  B := alpha*B*inv( A**T )
			//           or    B := alpha*B*inv( A**H ).
			if upper {
				for k = n; k >= 1; k-- {
					if nounit {
						if noconj {
							temp = one / a.Get(k-1, k-1)
						} else {
							temp = one / a.GetConj(k-1, k-1)
						}
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = a.Get(j-1, k-1)
							} else {
								temp = a.GetConj(j-1, k-1)
							}
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
						if noconj {
							temp = one / a.Get(k-1, k-1)
						} else {
							temp = one / a.GetConj(k-1, k-1)
						}
						for i = 1; i <= m; i++ {
							b.Set(i-1, k-1, temp*b.Get(i-1, k-1))
						}
					}
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = a.Get(j-1, k-1)
							} else {
								temp = a.GetConj(j-1, k-1)
							}
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

// Zherk performs one of the hermitian rank k operations
//
//    C := alpha*A*A**H + beta*C,
//
// or
//
//    C := alpha*A**H*A + beta*C,
//
// where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
// matrix and  A  is an  n by k  matrix in the  first case and a  k by n
// matrix in the second case.
func Zherk(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha float64, a *mat.CMatrix, lda int, beta float64, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var temp complex128
	var one, rtemp, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == Trans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldc < max(1, n) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zherk"), err)
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
						c.SetRe(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j-1; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.SetRe(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*A**H + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						c.SetRe(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j-1; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
				} else {
					c.SetRe(j-1, j-1, real(c.Get(j-1, j-1)))
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != complex(zero, 0) {
						temp = complex(alpha, 0) * a.GetConj(j-1, l-1)
						for i = 1; i <= j-1; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
						c.SetRe(j-1, j-1, real(c.GetReCmplx(j-1, j-1))+real(temp*a.Get(i-1, l-1)))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						c.SetRe(i-1, j-1, zero)
					}
				} else if beta != one {
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
				} else {
					c.SetRe(j-1, j-1, real(c.Get(j-1, j-1)))
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != complex(zero, 0) {
						temp = complex(alpha, 0) * a.GetConj(j-1, l-1)
						c.SetRe(j-1, j-1, real(c.Get(j-1, j-1))+real(temp*a.Get(j-1, l-1)))
						for i = j + 1; i <= n; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**H*A + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					temp = complex(zero, 0)
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, complex(alpha, 0)*temp)
					} else {
						c.Set(i-1, j-1, complex(alpha, 0)*temp+complex(beta, 0)*c.Get(i-1, j-1))
					}
				}
				rtemp = zero
				for l = 1; l <= k; l++ {
					rtemp += a.GetConjProd(l-1, j-1)
				}
				if beta == zero {
					c.SetRe(j-1, j-1, alpha*rtemp)
				} else {
					c.SetRe(j-1, j-1, alpha*rtemp+beta*real(c.Get(j-1, j-1)))
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				rtemp = zero
				for l = 1; l <= k; l++ {
					rtemp += a.GetConjProd(l-1, j-1)
				}
				if beta == zero {
					c.SetRe(j-1, j-1, alpha*rtemp)
				} else {
					c.SetRe(j-1, j-1, alpha*rtemp+beta*real(c.Get(j-1, j-1)))
				}
				for i = j + 1; i <= n; i++ {
					temp = complex(zero, 0)
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						c.Set(i-1, j-1, complex(alpha, 0)*temp)
					} else {
						c.Set(i-1, j-1, complex(alpha, 0)*temp+complex(beta, 0)*c.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Zsyrk performs one of the symmetric rank k operations
//
//    C := alpha*A*A**T + beta*C,
//
// or
//
//    C := alpha*A**T*A + beta*C,
//
// where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
// and  A  is an  n by k  matrix in the first case and a  k by n  matrix
// in the second case.
func Zsyrk(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha complex128, a *mat.CMatrix, lda int, beta complex128, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var one, temp, zero complex128
	var i, j, l, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == ConjTrans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldc < max(1, n) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zsyrk"), err)
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
	if trans == NoTrans {
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

// Zher2k performs one of the hermitian rank 2k operations
//
//    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
//
// or
//
//    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
//
// where  alpha and beta  are scalars with  beta  real,  C is an  n by n
// hermitian matrix and  A and B  are  n by k matrices in the first case
// and  k by n  matrices in the second case.
func Zher2k(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta float64, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var temp1, temp2, zero complex128
	var one float64
	var i, j, l, nrowa int

	one = 1.0
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == Trans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, nrowa) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	} else if ldc < max(1, n) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zher2k"), err)
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == real(zero) {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j-1; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
				}
			}
		} else {
			if beta == real(zero) {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*B**H + conjg( alpha )*B*A**H +
		//                   C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == real(zero) {
					for i = 1; i <= j; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j-1; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
				} else {
					c.Set(j-1, j-1, c.GetReCmplx(j-1, j-1))
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.GetConj(j-1, l-1)
						temp2 = cmplx.Conj(alpha * a.Get(j-1, l-1))
						for i = 1; i <= j-1; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
						c.SetRe(j-1, j-1, real(c.Get(j-1, j-1))+real(a.Get(j-1, l-1)*temp1+b.Get(j-1, l-1)*temp2))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == real(zero) {
					for i = j; i <= n; i++ {
						c.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j + 1; i <= n; i++ {
						c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1))
					}
					c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1)))
				} else {
					c.Set(j-1, j-1, c.GetReCmplx(j-1, j-1))
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.GetConj(j-1, l-1)
						temp2 = cmplx.Conj(alpha * a.Get(j-1, l-1))
						for i = j + 1; i <= n; i++ {
							c.Set(i-1, j-1, c.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
						c.SetRe(j-1, j-1, real(c.Get(j-1, j-1))+real(a.Get(j-1, l-1)*temp1+b.Get(j-1, l-1)*temp2))
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**H*B + conjg( alpha )*B**H*A +
		//                   C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.GetConj(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.GetConj(l-1, i-1)*a.Get(l-1, j-1)
					}
					if i == j {
						if beta == real(zero) {
							c.SetRe(j-1, j-1, real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						} else {
							c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1))+real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						}
					} else {
						if beta == real(zero) {
							c.Set(i-1, j-1, alpha*temp1+cmplx.Conj(alpha)*temp2)
						} else {
							c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1)+alpha*temp1+cmplx.Conj(alpha)*temp2)
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.GetConj(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.GetConj(l-1, i-1)*a.Get(l-1, j-1)
					}
					if i == j {
						if beta == real(zero) {
							c.SetRe(j-1, j-1, real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						} else {
							c.SetRe(j-1, j-1, beta*real(c.Get(j-1, j-1))+real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						}
					} else {
						if beta == real(zero) {
							c.Set(i-1, j-1, alpha*temp1+cmplx.Conj(alpha)*temp2)
						} else {
							c.Set(i-1, j-1, complex(beta, 0)*c.Get(i-1, j-1)+alpha*temp1+cmplx.Conj(alpha)*temp2)
						}
					}
				}
			}
		}
	}

	return
}

// Zsyr2k performs one of the symmetric rank 2k operations
//
//    C := alpha*A*B**T + alpha*B*A**T + beta*C,
//
// or
//
//    C := alpha*A**T*B + alpha*B**T*A + beta*C,
//
// where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
// and  A and B  are  n by k  matrices  in the  first  case  and  k by n
// matrices in the second case.
func Zsyr2k(uplo mat.MatUplo, trans mat.MatTrans, n, k int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta complex128, c *mat.CMatrix, ldc int) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, l, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == ConjTrans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if lda < max(1, nrowa) {
		err = fmt.Errorf("lda invalid: %v", lda)
	} else if ldb < max(1, nrowa) {
		err = fmt.Errorf("ldb invalid: %v", ldb)
	} else if ldc < max(1, n) {
		err = fmt.Errorf("ldc invalid: %v", ldc)
	}
	if err != nil {
		Xerbla2([]byte("Zsyr2k"), err)
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
	if trans == NoTrans {
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
						temp1 = temp1 + a.Get(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.Get(l-1, i-1)*a.Get(l-1, j-1)
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
						temp1 = temp1 + a.Get(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.Get(l-1, i-1)*a.Get(l-1, j-1)
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
