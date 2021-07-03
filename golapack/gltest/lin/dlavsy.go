package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlavsy performs one of the matrix-vector operations
//    x := A*x  or  x := A'*x,
// where x is an N element vector and A is one of the factors
// from the block U*D*U' or L*D*L' factorization computed by DSYTRF.
//
// If TRANS = 'N', multiplies by U  or U * D  (or L  or L * D)
// If TRANS = 'T', multiplies by U' or D * U' (or L' or D * L')
// If TRANS = 'C', multiplies by U' or D * U' (or L' or D * L')
func Dlavsy(uplo, trans, diag byte, n, nrhs *int, a *mat.Matrix, lda *int, ipiv *[]int, b *mat.Matrix, ldb *int, info *int) {
	var nounit bool
	var d11, d12, d21, d22, one, t1, t2 float64
	var j, k, kp int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if diag != 'U' && diag != 'N' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAVSY "), -(*info))
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	nounit = diag == 'N'
	//------------------------------------------
	//
	//     Compute  B := A * B  (No transpose)
	//
	//------------------------------------------
	if trans == 'N' {
		//        Compute  B := U*B
		//        where U = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
		if uplo == 'U' {
			//        Loop forward applying the transformations.
			k = 1
		label10:
			;
			if k > (*n) {
				goto label30
			}
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 pivot block
				//
				//              Multiply by the diagonal element if forming U * D.
				if nounit {
					goblas.Dscal(*nrhs, a.Get(k-1, k-1), b.Vector(k-1, 0), *ldb)
				}

				//              Multiply by  P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformation.
					err = goblas.Dger(k-1, *nrhs, one, a.Vector(0, k-1), 1, b.Vector(k-1, 0), *ldb, b, *ldb)

					//                 Interchange if P(K) .ne. I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}
				}
				k = k + 1
			} else {
				//              2 x 2 pivot block
				//
				//              Multiply by the diagonal block if forming U * D.
				if nounit {
					d11 = a.Get(k-1, k-1)
					d22 = a.Get(k+1-1, k+1-1)
					d12 = a.Get(k-1, k+1-1)
					d21 = d12
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1, j-1)
						t2 = b.Get(k+1-1, j-1)
						b.Set(k-1, j-1, d11*t1+d12*t2)
						b.Set(k+1-1, j-1, d21*t1+d22*t2)
					}
				}

				//              Multiply by  P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformations.
					err = goblas.Dger(k-1, *nrhs, one, a.Vector(0, k-1), 1, b.Vector(k-1, 0), *ldb, b, *ldb)
					err = goblas.Dger(k-1, *nrhs, one, a.Vector(0, k+1-1), 1, b.Vector(k+1-1, 0), *ldb, b, *ldb)

					//                 Interchange if P(K) .ne. I.
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}
				}
				k = k + 2
			}
			goto label10
		label30:

			//        Compute  B := L*B
			//        where L = P(1)*inv(L(1))* ... *P(m)*inv(L(m)) .
		} else {
			//           Loop backward applying the transformations to B.
			k = (*n)
		label40:
			;
			if k < 1 {
				goto label60
			}

			//           Test the pivot index.  If greater than zero, a 1 x 1
			//           pivot was used, otherwise a 2 x 2 pivot was used.
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 pivot block:
				//
				//              Multiply by the diagonal element if forming L * D.
				if nounit {
					goblas.Dscal(*nrhs, a.Get(k-1, k-1), b.Vector(k-1, 0), *ldb)
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != (*n) {
					kp = (*ipiv)[k-1]

					//                 Apply the transformation.
					err = goblas.Dger((*n)-k, *nrhs, one, a.Vector(k+1-1, k-1), 1, b.Vector(k-1, 0), *ldb, b.Off(k+1-1, 0), *ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}
				}
				k = k - 1

			} else {
				//              2 x 2 pivot block:
				//
				//              Multiply by the diagonal block if forming L * D.
				if nounit {
					d11 = a.Get(k-1-1, k-1-1)
					d22 = a.Get(k-1, k-1)
					d21 = a.Get(k-1, k-1-1)
					d12 = d21
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1-1, j-1)
						t2 = b.Get(k-1, j-1)
						b.Set(k-1-1, j-1, d11*t1+d12*t2)
						b.Set(k-1, j-1, d21*t1+d22*t2)
					}
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != (*n) {
					//                 Apply the transformation.
					err = goblas.Dger((*n)-k, *nrhs, one, a.Vector(k+1-1, k-1), 1, b.Vector(k-1, 0), *ldb, b.Off(k+1-1, 0), *ldb)
					err = goblas.Dger((*n)-k, *nrhs, one, a.Vector(k+1-1, k-1-1), 1, b.Vector(k-1-1, 0), *ldb, b.Off(k+1-1, 0), *ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}
				}
				k = k - 2
			}
			goto label40
		label60:
		}
		//----------------------------------------
		//
		//     Compute  B := A' * B  (transpose)
		//
		//----------------------------------------
	} else {
		//        Form  B := U'*B
		//        where U  = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
		//        and   U' = inv(U'(1))*P(1)* ... *inv(U'(m))*P(m)
		if uplo == 'U' {
			//           Loop backward applying the transformations.
			k = (*n)
		label70:
			;
			if k < 1 {
				goto label90
			}

			//           1 x 1 pivot block.
			if (*ipiv)[k-1] > 0 {
				if k > 1 {
					//                 Interchange if P(K) .ne. I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}

					//                 Apply the transformation
					err = goblas.Dgemv(mat.Trans, k-1, *nrhs, one, b, *ldb, a.Vector(0, k-1), 1, one, b.Vector(k-1, 0), *ldb)
				}
				if nounit {
					goblas.Dscal(*nrhs, a.Get(k-1, k-1), b.Vector(k-1, 0), *ldb)
				}
				k = k - 1

				//           2 x 2 pivot block.
			} else {
				if k > 2 {
					//                 Interchange if P(K) .ne. I.
					kp = absint((*ipiv)[k-1])
					if kp != k-1 {
						goblas.Dswap(*nrhs, b.Vector(k-1-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}

					//                 Apply the transformations
					err = goblas.Dgemv(mat.Trans, k-2, *nrhs, one, b, *ldb, a.Vector(0, k-1), 1, one, b.Vector(k-1, 0), *ldb)
					err = goblas.Dgemv(mat.Trans, k-2, *nrhs, one, b, *ldb, a.Vector(0, k-1-1), 1, one, b.Vector(k-1-1, 0), *ldb)
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(k-1-1, k-1-1)
					d22 = a.Get(k-1, k-1)
					d12 = a.Get(k-1-1, k-1)
					d21 = d12
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1-1, j-1)
						t2 = b.Get(k-1, j-1)
						b.Set(k-1-1, j-1, d11*t1+d12*t2)
						b.Set(k-1, j-1, d21*t1+d22*t2)
					}
				}
				k = k - 2
			}
			goto label70
		label90:

			//        Form  B := L'*B
			//        where L  = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
			//        and   L' = inv(L'(m))*P(m)* ... *inv(L'(1))*P(1)

		} else {
			//           Loop forward applying the L-transformations.
			k = 1
		label100:
			;
			if k > (*n) {
				goto label120
			}

			//           1 x 1 pivot block
			if (*ipiv)[k-1] > 0 {
				if k < (*n) {
					//                 Interchange if P(K) .ne. I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Dswap(*nrhs, b.Vector(k-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}

					//                 Apply the transformation
					err = goblas.Dgemv(mat.Trans, (*n)-k, *nrhs, one, b.Off(k+1-1, 0), *ldb, a.Vector(k+1-1, k-1), 1, one, b.Vector(k-1, 0), *ldb)
				}
				if nounit {
					goblas.Dscal(*nrhs, a.Get(k-1, k-1), b.Vector(k-1, 0), *ldb)
				}
				k = k + 1

				//           2 x 2 pivot block.
			} else {
				if k < (*n)-1 {
					//              Interchange if P(K) .ne. I.
					kp = absint((*ipiv)[k-1])
					if kp != k+1 {
						goblas.Dswap(*nrhs, b.Vector(k+1-1, 0), *ldb, b.Vector(kp-1, 0), *ldb)
					}

					//                 Apply the transformation
					err = goblas.Dgemv(mat.Trans, (*n)-k-1, *nrhs, one, b.Off(k+2-1, 0), *ldb, a.Vector(k+2-1, k+1-1), 1, one, b.Vector(k+1-1, 0), *ldb)
					err = goblas.Dgemv(mat.Trans, (*n)-k-1, *nrhs, one, b.Off(k+2-1, 0), *ldb, a.Vector(k+2-1, k-1), 1, one, b.Vector(k-1, 0), *ldb)
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(k-1, k-1)
					d22 = a.Get(k+1-1, k+1-1)
					d21 = a.Get(k+1-1, k-1)
					d12 = d21
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1, j-1)
						t2 = b.Get(k+1-1, j-1)
						b.Set(k-1, j-1, d11*t1+d12*t2)
						b.Set(k+1-1, j-1, d21*t1+d22*t2)
					}
				}
				k = k + 2
			}
			goto label100
		label120:
		}

	}
}
