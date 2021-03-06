package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zlavsp performs one of the matrix-vector operations
//       x := A*x  or  x := A^T*x,
//    where x is an N element vector and  A is one of the factors
//    from the symmetric factorization computed by ZSPTRF.
//    ZSPTRF produces a factorization of the form
//         U * D * U^T     or     L * D * L^T,
//    where U (or L) is a product of permutation and unit upper (lower)
//    triangular matrices, U^T (or L^T) is the transpose of
//    U (or L), and D is symmetric and block diagonal with 1 x 1 and
//    2 x 2 diagonal blocks.  The multipliers for the transformations
//    and the upper or lower triangular parts of the diagonal blocks
//    are stored columnwise in packed format in the linear array A.
//
//    If TRANS = 'N' or 'n', zlavspmultiplies either by U or U * D
//    (or L or L * D).
//    If TRANS = 'C' or 'c', zlavspmultiplies either by U^T or D * U^T
//    (or L^T or D * L^T ).
func zlavsp(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, a *mat.CVector, ipiv *[]int, b *mat.CMatrix) (err error) {
	var nounit bool
	var d11, d12, d21, d22, one, t1, t2 complex128
	var j, k, kc, kcnext, kp int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if trans != NoTrans && trans != Trans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans: trans=%s", trans)
	} else if diag != Unit && diag != NonUnit {
		err = fmt.Errorf("diag != Unit && diag != NonUnit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("zlavsp", err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit
	//------------------------------------------
	//
	//     Compute  B := A * B  (No transpose)
	//
	//------------------------------------------
	if trans == NoTrans {
		//        Compute  B := U*B
		//        where U = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
		if uplo == Upper {
			//        Loop forward applying the transformations.
			k = 1
			kc = 1
		label10:
			;
			if k > n {
				goto label30
			}

			//           1 x 1 pivot block
			if (*ipiv)[k-1] > 0 {
				//              Multiply by the diagonal element if forming U * D.
				if nounit {
					b.Off(k-1, 0).CVector().Scal(nrhs, a.Get(kc+k-1-1), b.Rows)
				}

				//              Multiply by P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformation.
					if err = b.Geru(k-1, nrhs, one, a.Off(kc-1), 1, b.Off(k-1, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}

					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
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
					d21 = d12
					for j = 1; j <= nrhs; j++ {
						t1 = b.Get(k-1, j-1)
						t2 = b.Get(k, j-1)
						b.Set(k-1, j-1, d11*t1+d12*t2)
						b.Set(k, j-1, d21*t1+d22*t2)
					}
				}

				//              Multiply by  P(K) * inv(U(K))  if K > 1.
				if k > 1 {
					//                 Apply the transformations.
					if err = b.Geru(k-1, nrhs, one, a.Off(kc-1), 1, b.Off(k-1, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}
					if err = b.Geru(k-1, nrhs, one, a.Off(kcnext-1), 1, b.Off(k, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}

					//                 Interchange if P(K) != I.
					kp = abs((*ipiv)[k-1])
					if kp != k {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
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
			k = n
			kc = n*(n+1)/2 + 1
		label40:
			;
			if k < 1 {
				goto label60
			}
			kc = kc - (n - k + 1)

			//           Test the pivot index.  If greater than zero, a 1 x 1
			//           pivot was used, otherwise a 2 x 2 pivot was used.
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 pivot block:
				//
				//              Multiply by the diagonal element if forming L * D.
				if nounit {
					b.Off(k-1, 0).CVector().Scal(nrhs, a.Get(kc-1), b.Rows)
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != n {
					kp = (*ipiv)[k-1]

					//                 Apply the transformation.
					if err = b.Off(k, 0).Geru(n-k, nrhs, one, a.Off(kc), 1, b.Off(k-1, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					if kp != k {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
					}
				}
				k = k - 1

			} else {
				//              2 x 2 pivot block:
				kcnext = kc - (n - k + 2)

				//              Multiply by the diagonal block if forming L * D.
				if nounit {
					d11 = a.Get(kcnext - 1)
					d22 = a.Get(kc - 1)
					d21 = a.Get(kcnext + 1 - 1)
					d12 = d21
					for j = 1; j <= nrhs; j++ {
						t1 = b.Get(k-1-1, j-1)
						t2 = b.Get(k-1, j-1)
						b.Set(k-1-1, j-1, d11*t1+d12*t2)
						b.Set(k-1, j-1, d21*t1+d22*t2)
					}
				}

				//              Multiply by  P(K) * inv(L(K))  if K < N.
				if k != n {
					//                 Apply the transformation.
					if err = b.Off(k, 0).Geru(n-k, nrhs, one, a.Off(kc), 1, b.Off(k-1, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}
					if err = b.Off(k, 0).Geru(n-k, nrhs, one, a.Off(kcnext+2-1), 1, b.Off(k-1-1, 0).CVector(), b.Rows); err != nil {
						panic(err)
					}

					//                 Interchange if a permutation was applied at the
					//                 K-th step of the factorization.
					kp = abs((*ipiv)[k-1])
					if kp != k {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
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
		//     Compute  B := A^T * B  (transpose)
		//
		//-------------------------------------------------
	} else {
		//        Form  B := U^T*B
		//        where U  = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
		//        and   U^T = inv(U^T(1))*P(1)* ... *inv(U^T(m))*P(m)
		if uplo == Upper {
			//           Loop backward applying the transformations.
			k = n
			kc = n*(n+1)/2 + 1
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
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
					}

					//                 Apply the transformation:
					//                    y := y - B' * conjg(x)
					//                 where x is a column of A and y is a row of B.
					if err = b.Off(k-1, 0).CVector().Gemv(Trans, k-1, nrhs, one, b, a.Off(kc-1), 1, one, b.Rows); err != nil {
						panic(err)
					}
				}
				if nounit {
					b.Off(k-1, 0).CVector().Scal(nrhs, a.Get(kc+k-1-1), b.Rows)
				}
				k = k - 1

				//           2 x 2 pivot block.
			} else {
				kcnext = kc - (k - 1)
				if k > 2 {
					//                 Interchange if P(K) != I.
					kp = abs((*ipiv)[k-1])
					if kp != k-1 {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1-1, 0).CVector(), b.Rows, b.Rows)
					}

					//                 Apply the transformations.
					if err = b.Off(k-1, 0).CVector().Gemv(Trans, k-2, nrhs, one, b, a.Off(kc-1), 1, one, b.Rows); err != nil {
						panic(err)
					}

					if err = b.Off(k-1-1, 0).CVector().Gemv(Trans, k-2, nrhs, one, b, a.Off(kcnext-1), 1, one, b.Rows); err != nil {
						panic(err)
					}
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(kc - 1 - 1)
					d22 = a.Get(kc + k - 1 - 1)
					d12 = a.Get(kc + k - 2 - 1)
					d21 = d12
					for j = 1; j <= nrhs; j++ {
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

			//        Form  B := L^T*B
			//        where L  = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
			//        and   L^T = inv(L(m))*P(m)* ... *inv(L(1))*P(1)
		} else {
			//           Loop forward applying the L-transformations.
			k = 1
			kc = 1
		label100:
			;
			if k > n {
				goto label120
			}

			//           1 x 1 pivot block
			if (*ipiv)[k-1] > 0 {
				if k < n {
					//                 Interchange if P(K) != I.
					kp = (*ipiv)[k-1]
					if kp != k {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
					}

					//                 Apply the transformation
					if err = b.Off(k-1, 0).CVector().Gemv(Trans, n-k, nrhs, one, b.Off(k, 0), a.Off(kc), 1, one, b.Rows); err != nil {
						panic(err)
					}
				}
				if nounit {
					b.Off(k-1, 0).CVector().Scal(nrhs, a.Get(kc-1), b.Rows)
				}
				kc = kc + n - k + 1
				k = k + 1

				//           2 x 2 pivot block.
			} else {
				kcnext = kc + n - k + 1
				if k < n-1 {
					//              Interchange if P(K) != I.
					kp = abs((*ipiv)[k-1])
					if kp != k+1 {
						b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k, 0).CVector(), b.Rows, b.Rows)
					}

					//                 Apply the transformation
					if err = b.Off(k, 0).CVector().Gemv(Trans, n-k-1, nrhs, one, b.Off(k+2-1, 0), a.Off(kcnext), 1, one, b.Rows); err != nil {
						panic(err)
					}

					if err = b.Off(k-1, 0).CVector().Gemv(Trans, n-k-1, nrhs, one, b.Off(k+2-1, 0), a.Off(kc+2-1), 1, one, b.Rows); err != nil {
						panic(err)
					}
				}

				//              Multiply by the diagonal block if non-unit.
				if nounit {
					d11 = a.Get(kc - 1)
					d22 = a.Get(kcnext - 1)
					d21 = a.Get(kc + 1 - 1)
					d12 = d21
					for j = 1; j <= nrhs; j++ {
						t1 = b.Get(k-1, j-1)
						t2 = b.Get(k, j-1)
						b.Set(k-1, j-1, d11*t1+d12*t2)
						b.Set(k, j-1, d21*t1+d22*t2)
					}
				}
				kc = kcnext + (n - k)
				k = k + 2
			}
			goto label100
		label120:
		}

	}

	return
}
