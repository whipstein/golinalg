package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math/cmplx"
)

// Zlavhp performs one of the matrix-vector operations
//       x := A*x  or  x := A^H*x,
//    where x is an N element vector and  A is one of the factors
//    from the symmetric factorization computed by ZHPTRF.
//    ZHPTRF produces a factorization of the form
//         U * D * U^H     or     L * D * L^H,
//    where U (or L) is a product of permutation and unit upper (lower)
//    triangular matrices, U^H (or L^H) is the conjugate transpose of
//    U (or L), and D is Hermitian and block diagonal with 1 x 1 and
//    2 x 2 diagonal blocks.  The multipliers for the transformations
//    and the upper or lower triangular parts of the diagonal blocks
//    are stored columnwise in packed format in the linear array A.
//
//    If TRANS = 'N' or 'n', ZLAVHP multiplies either by U or U * D
//    (or L or L * D).
//    If TRANS = 'C' or 'c', ZLAVHP multiplies either by U^H or D * U^H
//    (or L^H or D * L^H ).
func Zlavhp(uplo, trans, diag byte, n, nrhs *int, a *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var nounit bool
	var d11, d12, d21, d22, one, t1, t2 complex128
	var j, k, kc, kcnext, kp int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'C' {
		(*info) = -2
	} else if diag != 'U' && diag != 'N' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAVHP "), -(*info))
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
			kc = 1
		label10:
			;
			if k > (*n) {
				goto label30
			}

			//           1 x 1 pivot block
			if (*ipiv)[k-1] > 0 {
				//              Multiply by the diagonal element if forming U * D.
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(kc+k-1-1), b.CVector(k-1, 0), ldb)
				}

				//              Multiply by P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformation.
					goblas.Zgeru(toPtr(k-1), nrhs, &one, a.Off(kc-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)

					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}
				}
				kc = kc + k
				k = k + 1
			} else {
				//              2 x 2 pivot block
				kcnext = kc + k

				//              Multiply by the diagonal block if forming U * D.
				if nounit {
					d11 = a.Get(kcnext - 1 - 1)
					d22 = a.Get(kcnext + k - 1)
					d12 = a.Get(kcnext + k - 1 - 1)
					d21 = cmplx.Conj(d12)
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
					goblas.Zgeru(toPtr(k-1), nrhs, &one, a.Off(kc-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)
					goblas.Zgeru(toPtr(k-1), nrhs, &one, a.Off(kcnext-1), func() *int { y := 1; return &y }(), b.CVector(k+1-1, 0), ldb, b, ldb)

					//                 Interchange if P(K) != I.
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}
				}
				kc = kcnext + k + 1
				k = k + 2
			}
			goto label10
		label30:

			//        Compute  B := L*B
			//        where L = P(1)*inv(L(1))* ... *P(m)*inv(L(m)) .
		} else {
			//           Loop backward applying the transformations to B.
			k = (*n)
			kc = (*n)*((*n)+1)/2 + 1
		label40:
			;
			if k < 1 {
				goto label60
			}
			kc = kc - ((*n) - k + 1)

			//           Test the pivot index.  If greater than zero, a 1 x 1
			//           pivot was used, otherwise a 2 x 2 pivot was used.
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 pivot block:
				//
				//              Multiply by the diagonal element if forming L * D.
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(kc-1), b.CVector(k-1, 0), ldb)
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != (*n) {
					kp = (*ipiv)[k-1]

					//                 Apply the transformation.
					goblas.Zgeru(toPtr((*n)-k), nrhs, &one, a.Off(kc+1-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+1-1, 0), ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}
				}
				k = k - 1

			} else {
				//              2 x 2 pivot block:
				kcnext = kc - ((*n) - k + 2)

				//              Multiply by the diagonal block if forming L * D.
				if nounit {
					d11 = a.Get(kcnext - 1)
					d22 = a.Get(kc - 1)
					d21 = a.Get(kcnext + 1 - 1)
					d12 = cmplx.Conj(d21)
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
					goblas.Zgeru(toPtr((*n)-k), nrhs, &one, a.Off(kc+1-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+1-1, 0), ldb)
					goblas.Zgeru(toPtr((*n)-k), nrhs, &one, a.Off(kcnext+2-1), func() *int { y := 1; return &y }(), b.CVector(k-1-1, 0), ldb, b.Off(k+1-1, 0), ldb)

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					kp = absint((*ipiv)[k-1])
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}
				}
				kc = kcnext
				k = k - 2
			}
			goto label40
		label60:
		}
		//-------------------------------------------------
		//
		//     Compute  B := A^H * B  (conjugate transpose)
		//
		//-------------------------------------------------
	} else {
		//        Form  B := U^H*B
		//        where U  = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
		//        and   U^H = inv(U^H(1))*P(1)* ... *inv(U^H(m))*P(m)
		if uplo == 'U' {
			//           Loop backward applying the transformations.
			k = (*n)
			kc = (*n)*((*n)+1)/2 + 1
		label70:
			;
			if k < 1 {
				goto label90
			}
			kc = kc - k

			//           1 x 1 pivot block.
			if (*ipiv)[k-1] > 0 {
				if k > 1 {
					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformation:
					//                    y := y - B' * conjg(x)
					//                 where x is a column of A and y is a row of B.
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr(k-1), nrhs, &one, b, ldb, a.Off(kc-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				}
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(kc+k-1-1), b.CVector(k-1, 0), ldb)
				}
				k = k - 1

				//           2 x 2 pivot block.
			} else {
				kcnext = kc - (k - 1)
				if k > 2 {
					//                 Interchange if P(K) != I.
					kp = absint((*ipiv)[k-1])
					if kp != k-1 {
						goblas.Zswap(nrhs, b.CVector(k-1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformations.
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr(k-2), nrhs, &one, b, ldb, a.Off(kc-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)

					golapack.Zlacgv(nrhs, b.CVector(k-1-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr(k-2), nrhs, &one, b, ldb, a.Off(kcnext-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k-1-1, 0), ldb)
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(kc - 1 - 1)
					d22 = a.Get(kc + k - 1 - 1)
					d12 = a.Get(kc + k - 2 - 1)
					d21 = cmplx.Conj(d12)
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1-1, j-1)
						t2 = b.Get(k-1, j-1)
						b.Set(k-1-1, j-1, d11*t1+d12*t2)
						b.Set(k-1, j-1, d21*t1+d22*t2)
					}
				}
				kc = kcnext
				k = k - 2
			}
			goto label70
		label90:

			//        Form  B := L^H*B
			//        where L  = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
			//        and   L^H = inv(L(m))*P(m)* ... *inv(L(1))*P(1)
		} else {
			//           Loop forward applying the L-transformations.
			k = 1
			kc = 1
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
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr((*n)-k), nrhs, &one, b.Off(k+1-1, 0), ldb, a.Off(kc+1-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				}
				if nounit {
					goblas.Zscal(nrhs, a.GetPtr(kc-1), b.CVector(k-1, 0), ldb)
				}
				kc = kc + (*n) - k + 1
				k = k + 1

				//           2 x 2 pivot block.
			} else {
				kcnext = kc + (*n) - k + 1
				if k < (*n)-1 {
					//              Interchange if P(K) != I.
					kp = absint((*ipiv)[k-1])
					if kp != k+1 {
						goblas.Zswap(nrhs, b.CVector(k+1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
					}

					//                 Apply the transformation
					golapack.Zlacgv(nrhs, b.CVector(k+1-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr((*n)-k-1), nrhs, &one, b.Off(k+2-1, 0), ldb, a.Off(kcnext+1-1), func() *int { y := 1; return &y }(), &one, b.CVector(k+1-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k+1-1, 0), ldb)

					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
					goblas.Zgemv(ConjTrans, toPtr((*n)-k-1), nrhs, &one, b.Off(k+2-1, 0), ldb, a.Off(kc+2-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
					golapack.Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(kc - 1)
					d22 = a.Get(kcnext - 1)
					d21 = a.Get(kc + 1 - 1)
					d12 = cmplx.Conj(d21)
					for j = 1; j <= (*nrhs); j++ {
						t1 = b.Get(k-1, j-1)
						t2 = b.Get(k+1-1, j-1)
						b.Set(k-1, j-1, d11*t1+d12*t2)
						b.Set(k+1-1, j-1, d21*t1+d22*t2)
					}
				}
				kc = kcnext + ((*n) - k)
				k = k + 2
			}
			goto label100
		label120:
		}

	}
}
