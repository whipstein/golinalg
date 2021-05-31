package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrfaa2stage computes the factorization of a complex symmetric matrix A
// using the Aasen's algorithm.  The form of the factorization is
//
//    A = U**T*T*U  or  A = L*T*L**T
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is a complex symmetric band matrix with the
// bandwidth of NB (NB is internally selected and stored in TB( 1 ), and T is
// LU factorized with partial pivoting).
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Zsytrfaa2stage(uplo byte, n *int, a *mat.CMatrix, lda *int, tb *mat.CVector, ltb *int, ipiv, ipiv2 *[]int, work *mat.CVector, lwork, info *int) {
	var tquery, upper, wquery bool
	var cone, czero, piv complex128
	var i, i1, i2, iinfo, j, jb, k, kb, ldtb, nb, nt, td int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	wquery = ((*lwork) == -1)
	tquery = ((*ltb) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*ltb) < 4*(*n) && !tquery {
		(*info) = -6
	} else if (*lwork) < (*n) && !wquery {
		(*info) = -10
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYTRF_AA_2STAGE"), -(*info))
		return
	}

	//     Answer the query
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZSYTRF_AA_2STAGE"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if (*info) == 0 {
		if tquery {
			tb.SetRe(0, float64((3*nb+1)*(*n)))
		}
		if wquery {
			work.SetRe(0, float64((*n)*nb))
		}
	}
	if tquery || wquery {
		return
	}

	//     Quick return
	if (*n) == 0 {
		return
	}

	//     Determine the number of the block size
	ldtb = (*ltb) / (*n)
	if ldtb < 3*nb+1 {
		nb = (ldtb - 1) / 3
	}
	if (*lwork) < nb*(*n) {
		nb = (*lwork) / (*n)
	}

	//     Determine the number of the block columns
	nt = ((*n) + nb - 1) / nb
	td = 2 * nb
	kb = minint(nb, *n)

	//     Initialize vectors/matrices
	for j = 1; j <= kb; j++ {
		(*ipiv)[j-1] = j
	}

	//     Save NB
	tb.SetRe(0, float64(nb))

	if upper {
		//        .....................................................
		//        Factorize A as U**T*D*U using the upper triangle of A
		//        .....................................................
		for j = 0; j <= nt-1; j++ {
			//           Generate Jth column of W and H
			kb = minint(nb, (*n)-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                  H(I,J) = T(I,I)*U(I,J) + T(I+1,I)*U(I+1,J)
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					goblas.Zgemm(NoTrans, NoTrans, &nb, &kb, &jb, &cone, tb.CMatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((i-1)*nb+1-1, j*nb+1-1), lda, &czero, work.CMatrixOff(i*nb+1-1, *n, opts), n)
				} else {
					//                 H(I,J) = T(I,I-1)*U(I-1,J) + T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == (j - 1) {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					goblas.Zgemm(NoTrans, NoTrans, &nb, &kb, &jb, &cone, tb.CMatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((i-2)*nb+1-1, j*nb+1-1), lda, &czero, work.CMatrixOff(i*nb+1-1, *n, opts), n)
				}
			}

			//           Compute T(J,J)
			Zlacpy('U', &kb, &kb, a.Off(j*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = U(1:J,J)'*H(1:J)
				goblas.Zgemm(Trans, NoTrans, &kb, &kb, toPtr((j-1)*nb), toPtrc128(-cone), a.Off(0, j*nb+1-1), lda, work.CMatrixOff(nb+1-1, *n, opts), n, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				//              T(J,J) += U(J,J)'*T(J,J-1)*U(J-1,J)
				goblas.Zgemm(Trans, NoTrans, &kb, &nb, &kb, &cone, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), &czero, work.CMatrix(*n, opts), n)
				goblas.Zgemm(NoTrans, NoTrans, &kb, &kb, &nb, toPtrc128(-cone), work.CMatrix(*n, opts), n, a.Off((j-2)*nb+1-1, j*nb+1-1), lda, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}

			//           Expand T(J,J) into full format
			for i = 1; i <= kb; i++ {
				for k = i + 1; k <= kb; k++ {
					tb.Set(td+(k-i)+1+(j*nb+i-1)*ldtb-1, tb.Get(td-(k-(i+1))+(j*nb+k-1)*ldtb-1))
				}
			}
			if j > 0 {
				//c               CALL CHEGST( 1, 'Upper', KB,
				//c     $                      TB( TD+1 + (J*NB)*LDTB ), LDTB-1,
				//c     $                      A( (J-1)*NB+1, J*NB+1 ), LDA, IINFO )
				goblas.Ztrsm(Left, Upper, Trans, NonUnit, &kb, &kb, &cone, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				goblas.Ztrsm(Right, Upper, NoTrans, NonUnit, &kb, &kb, &cone, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}

			if j < nt-1 {
				if j > 0 {
					//                 Compute H(J,J)
					if j == 1 {
						goblas.Zgemm(NoTrans, NoTrans, &kb, &kb, &kb, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((j-1)*nb+1-1, j*nb+1-1), lda, &czero, work.CMatrixOff(j*nb+1-1, *n, opts), n)
					} else {
						goblas.Zgemm(NoTrans, NoTrans, &kb, &kb, toPtr(nb+kb), &cone, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((j-2)*nb+1-1, j*nb+1-1), lda, &czero, work.CMatrixOff(j*nb+1-1, *n, opts), n)
					}

					//                 Update with the previous column
					goblas.Zgemm(Trans, NoTrans, &nb, toPtr((*n)-(j+1)*nb), toPtr(j*nb), toPtrc128(-cone), work.CMatrixOff(nb+1-1, *n, opts), n, a.Off(0, (j+1)*nb+1-1), lda, &cone, a.Off(j*nb+1-1, (j+1)*nb+1-1), lda)
				}

				//              Copy panel to workspace to call ZGETRF
				for k = 1; k <= nb; k++ {
					goblas.Zcopy(toPtr((*n)-(j+1)*nb), a.CVector(j*nb+k-1, (j+1)*nb+1-1), lda, work.Off(1+(k-1)*(*n)-1), func() *int { y := 1; return &y }())
				}

				//              Factorize panel
				Zgetrf(toPtr((*n)-(j+1)*nb), &nb, work.CMatrix(*n, opts), n, toSlice(ipiv, (j+1)*nb+1-1), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Copy panel back
				for k = 1; k <= nb; k++ {
					goblas.Zcopy(toPtr((*n)-(j+1)*nb), work.Off(1+(k-1)*(*n)-1), func() *int { y := 1; return &y }(), a.CVector(j*nb+k-1, (j+1)*nb+1-1), lda)
				}

				//              Compute T(J+1, J), zero out for GEMM update
				kb = minint(nb, (*n)-(j+1)*nb)
				Zlaset('F', &kb, &nb, &czero, &czero, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Zlacpy('U', &kb, &nb, work.CMatrix(*n, opts), n, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					goblas.Ztrsm(Right, Upper, NoTrans, Unit, &kb, &nb, &cone, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				}

				//              Copy T(J,J+1) into T(J+1, J), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Zlaset('L', &kb, &nb, &czero, &cone, a.Off(j*nb+1-1, (j+1)*nb+1-1), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Zswap(toPtr(k-1), a.CVector((j+1)*nb+1-1, i1-1), func() *int { y := 1; return &y }(), a.CVector((j+1)*nb+1-1, i2-1), func() *int { y := 1; return &y }())
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Zswap(toPtr(i2-i1-1), a.CVector(i1-1, i1+1-1), lda, a.CVector(i1+1-1, i2-1), func() *int { y := 1; return &y }())
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Zswap(toPtr((*n)-i2), a.CVector(i1-1, i2+1-1), lda, a.CVector(i2-1, i2+1-1), lda)
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Zswap(toPtr(j*nb), a.CVector(0, i1-1), func() *int { y := 1; return &y }(), a.CVector(0, i2-1), func() *int { y := 1; return &y }())
						}
					}
				}
			}
		}
	} else {
		//        .....................................................
		//        Factorize A as L*D*L**T using the lower triangle of A
		//        .....................................................
		for j = 0; j <= nt-1; j++ {
			//           Generate Jth column of W and H
			kb = minint(nb, (*n)-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                  H(I,J) = T(I,I)*L(J,I)' + T(I+1,I)'*L(J,I+1)'
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					goblas.Zgemm(NoTrans, Trans, &nb, &kb, &jb, &cone, tb.CMatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (i-1)*nb+1-1), lda, &czero, work.CMatrixOff(i*nb+1-1, *n, opts), n)
				} else {
					//                 H(I,J) = T(I,I-1)*L(J,I-1)' + T(I,I)*L(J,I)' + T(I,I+1)*L(J,I+1)'
					if i == (j - 1) {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					goblas.Zgemm(NoTrans, Trans, &nb, &kb, &jb, &cone, tb.CMatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (i-2)*nb+1-1), lda, &czero, work.CMatrixOff(i*nb+1-1, *n, opts), n)
				}
			}

			//           Compute T(J,J)
			Zlacpy('L', &kb, &kb, a.Off(j*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = L(J,1:J)*H(1:J)
				goblas.Zgemm(NoTrans, NoTrans, &kb, &kb, toPtr((j-1)*nb), toPtrc128(-cone), a.Off(j*nb+1-1, 0), lda, work.CMatrixOff(nb+1-1, *n, opts), n, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				//              T(J,J) += L(J,J)*T(J,J-1)*L(J,J-1)'
				goblas.Zgemm(NoTrans, NoTrans, &kb, &nb, &kb, &cone, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), &czero, work.CMatrix(*n, opts), n)
				goblas.Zgemm(NoTrans, Trans, &kb, &kb, &nb, toPtrc128(-cone), work.CMatrix(*n, opts), n, a.Off(j*nb+1-1, (j-2)*nb+1-1), lda, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}

			//           Expand T(J,J) into full format
			for i = 1; i <= kb; i++ {
				for k = i + 1; k <= kb; k++ {
					tb.Set(td-(k-(i+1))+(j*nb+k-1)*ldtb-1, tb.Get(td+(k-i)+1+(j*nb+i-1)*ldtb-1))
				}
			}
			if j > 0 {
				//c               CALL CHEGST( 1, 'Lower', KB,
				//c     $                      TB( TD+1 + (J*NB)*LDTB ), LDTB-1,
				//c     $                      A( J*NB+1, (J-1)*NB+1 ), LDA, IINFO )
				goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, &kb, &kb, &cone, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				goblas.Ztrsm(Right, Lower, Trans, NonUnit, &kb, &kb, &cone, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}

			//           Symmetrize T(J,J)
			for i = 1; i <= kb; i++ {
				for k = i + 1; k <= kb; k++ {
					tb.Set(td-(k-(i+1))+(j*nb+k-1)*ldtb-1, tb.Get(td+(k-i)+1+(j*nb+i-1)*ldtb-1))
				}
			}

			if j < nt-1 {
				if j > 0 {
					//                 Compute H(J,J)
					if j == 1 {
						goblas.Zgemm(NoTrans, Trans, &kb, &kb, &kb, &cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, &czero, work.CMatrixOff(j*nb+1-1, *n, opts), n)
					} else {
						goblas.Zgemm(NoTrans, Trans, &kb, &kb, toPtr(nb+kb), &cone, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (j-2)*nb+1-1), lda, &czero, work.CMatrixOff(j*nb+1-1, *n, opts), n)
					}

					//                 Update with the previous column
					goblas.Zgemm(NoTrans, NoTrans, toPtr((*n)-(j+1)*nb), &nb, toPtr(j*nb), toPtrc128(-cone), a.Off((j+1)*nb+1-1, 0), lda, work.CMatrixOff(nb+1-1, *n, opts), n, &cone, a.Off((j+1)*nb+1-1, j*nb+1-1), lda)
				}

				//              Factorize panel
				Zgetrf(toPtr((*n)-(j+1)*nb), &nb, a.Off((j+1)*nb+1-1, j*nb+1-1), lda, toSlice(ipiv, (j+1)*nb+1-1), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Compute T(J+1, J), zero out for GEMM update
				kb = minint(nb, (*n)-(j+1)*nb)
				Zlaset('F', &kb, &nb, &czero, &czero, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Zlacpy('U', &kb, &nb, a.Off((j+1)*nb+1-1, j*nb+1-1), lda, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					goblas.Ztrsm(Right, Lower, Trans, Unit, &kb, &nb, &cone, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				}

				//              Copy T(J+1,J) into T(J, J+1), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Zlaset('U', &kb, &nb, &czero, &cone, a.Off((j+1)*nb+1-1, j*nb+1-1), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Zswap(toPtr(k-1), a.CVector(i1-1, (j+1)*nb+1-1), lda, a.CVector(i2-1, (j+1)*nb+1-1), lda)
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Zswap(toPtr(i2-i1-1), a.CVector(i1+1-1, i1-1), func() *int { y := 1; return &y }(), a.CVector(i2-1, i1+1-1), lda)
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Zswap(toPtr((*n)-i2), a.CVector(i2+1-1, i1-1), func() *int { y := 1; return &y }(), a.CVector(i2+1-1, i2-1), func() *int { y := 1; return &y }())
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Zswap(toPtr(j*nb), a.CVector(i1-1, 0), lda, a.CVector(i2-1, 0), lda)
						}
					}
				}

				//              Apply pivots to previous columns of L
				//
				//c               CALL ZLASWP( J*NB, A( 1, 1 ), LDA,
				//c     $                     (J+1)*NB+1, (J+1)*NB+KB, IPIV, 1 )
			}
		}
	}

	//     Factor the band matrix
	Zgbtrf(n, n, &nb, &nb, tb.CMatrix(ldtb, opts), &ldtb, ipiv2, info)
}
