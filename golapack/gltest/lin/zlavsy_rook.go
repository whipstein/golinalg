package lin

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlavsyrook performs one of the matrix-vector operations
//    x := A*x  or  x := A'*x,
// where x is an N element vector and  A is one of the factors
// from the block U*D*U' or L*D*L' factorization computed by ZSYTRF_ROOK.
//
// If TRANS = 'N', multiplies by U  or U * D  (or L  or L * D)
// If TRANS = 'T', multiplies by U' or D * U' (or L' or D * L')
func Zlavsyrook(uplo, trans, diag byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var nounit bool
	var cone, d11, d12, d21, d22, t1, t2 complex128
	var j, k, kp int

	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' {
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
		gltest.Xerbla([]byte("ZLAVSY_ROOK "), -(*info))
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
					goblas.Zscal(nrhs, a.GetPtr(k-1, k-1), b.CVector(k-1, 0), ldb)
				}

				//              Multiply by  P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformation.
					goblas.Zgeru(toPtr(k-1), nrhs, &cone, a.CVector(0, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)

					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
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
					goblas.Zgeru(toPtr(k-1), nrhs, &cone, a.CVector(0, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)
					goblas.Zgeru(toPtr(k-1), nrhs, &cone, a.CVector(0, k+1-1), func() *int { y := 1; return &y }(), b.CVector(k+1-1, 0), ldb, b, ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					//
					//                 Swap the first of pair with IMAXth
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 NOW swap the first of pair with Pth
					kp = absint((*ipiv)[k+1-1])
					if kp != k+1 {
						goblas.Zswap(nrhs, b.CVector(k+1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
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
					goblas.Zscal(nrhs, a.GetPtr(k-1, k-1), b.CVector(k-1, 0), ldb)
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != (*n) {
					kp = (*ipiv)[k-1]

					//                 Apply the transformation.
					goblas.Zgeru(toPtr((*n)-k), nrhs, &cone, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+1-1, 0), ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
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
					goblas.Zgeru(toPtr((*n)-k), nrhs, &cone, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+1-1, 0), ldb)
					goblas.Zgeru(toPtr((*n)-k), nrhs, &cone, a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }(), b.CVector(k-1-1, 0), ldb, b.Off(k+1-1, 0), ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					//
					//                 Swap the second of pair with IMAXth
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 NOW swap the first of pair with Pth
					kp = absint((*ipiv)[k-1-1])
					if kp != k-1 {
						goblas.Zswap(nrhs, b.CVector(k-1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
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
	} else if trans == 'T' {
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
					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformation
					goblas.Zgemv(Trans, toPtr(k-1), nrhs, &cone, b, ldb, a.CVector(0, k-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k-1, 0), ldb)
				}
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(k-1, k-1), b.CVector(k-1, 0), ldb)
				}
				k = k - 1

				//           2 x 2 pivot block.
			} else {
				if k > 2 {
					//                 Swap the second of pair with Pth
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Now swap the first of pair with IMAX(r)th
					kp = absint((*ipiv)[k-1-1])
					if kp != k-1 {
						goblas.Zswap(nrhs, b.CVector(k-1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformations
					goblas.Zgemv(Trans, toPtr(k-2), nrhs, &cone, b, ldb, a.CVector(0, k-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k-1, 0), ldb)
					goblas.Zgemv(Trans, toPtr(k-2), nrhs, &cone, b, ldb, a.CVector(0, k-1-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k-1-1, 0), ldb)
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
					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformation
					goblas.Zgemv(Trans, toPtr((*n)-k), nrhs, &cone, b.Off(k+1-1, 0), ldb, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k-1, 0), ldb)
				}
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(k-1, k-1), b.CVector(k-1, 0), ldb)
				}
				k = k + 1

				//           2 x 2 pivot block.
			} else {
				if k < (*n)-1 {
					//                 Swap the first of pair with Pth
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Now swap the second of pair with IMAX(r)th
					kp = absint((*ipiv)[k+1-1])
					if kp != k+1 {
						goblas.Zswap(nrhs, b.CVector(k+1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformation
					goblas.Zgemv(Trans, toPtr((*n)-k-1), nrhs, &cone, b.Off(k+2-1, 0), ldb, a.CVector(k+2-1, k+1-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k+1-1, 0), ldb)
					goblas.Zgemv(Trans, toPtr((*n)-k-1), nrhs, &cone, b.Off(k+2-1, 0), ldb, a.CVector(k+2-1, k-1), func() *int { y := 1; return &y }(), &cone, b.CVector(k-1, 0), ldb)
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
