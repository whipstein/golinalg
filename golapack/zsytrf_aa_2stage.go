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
	var err error
	_ = err

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
	} else if (*lda) < max(1, *n) {
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
	kb = min(nb, *n)

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
			kb = min(nb, (*n)-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                  H(I,J) = T(I,I)*U(I,J) + T(I+1,I)*U(I+1,J)
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					err = goblas.Zgemm(NoTrans, NoTrans, nb, kb, jb, cone, tb.CMatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), a.Off((i-1)*nb, j*nb), czero, work.CMatrixOff(i*nb, *n, opts))
				} else {
					//                 H(I,J) = T(I,I-1)*U(I-1,J) + T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == (j - 1) {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					err = goblas.Zgemm(NoTrans, NoTrans, nb, kb, jb, cone, tb.CMatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), a.Off((i-2)*nb, j*nb), czero, work.CMatrixOff(i*nb, *n, opts))
				}
			}

			//           Compute T(J,J)
			Zlacpy('U', &kb, &kb, a.Off(j*nb, j*nb), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = U(1:J,J)'*H(1:J)
				err = goblas.Zgemm(Trans, NoTrans, kb, kb, (j-1)*nb, -cone, a.Off(0, j*nb), work.CMatrixOff(nb, *n, opts), cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
				//              T(J,J) += U(J,J)'*T(J,J-1)*U(J-1,J)
				err = goblas.Zgemm(Trans, NoTrans, kb, nb, kb, cone, a.Off((j-1)*nb, j*nb), tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), czero, work.CMatrix(*n, opts))
				err = goblas.Zgemm(NoTrans, NoTrans, kb, kb, nb, -cone, work.CMatrix(*n, opts), a.Off((j-2)*nb, j*nb), cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
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
				err = goblas.Ztrsm(Left, Upper, Trans, NonUnit, kb, kb, cone, a.Off((j-1)*nb, j*nb), tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
				err = goblas.Ztrsm(Right, Upper, NoTrans, NonUnit, kb, kb, cone, a.Off((j-1)*nb, j*nb), tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
			}

			if j < nt-1 {
				if j > 0 {
					//                 Compute H(J,J)
					if j == 1 {
						err = goblas.Zgemm(NoTrans, NoTrans, kb, kb, kb, cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), a.Off((j-1)*nb, j*nb), czero, work.CMatrixOff(j*nb, *n, opts))
					} else {
						err = goblas.Zgemm(NoTrans, NoTrans, kb, kb, nb+kb, cone, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), a.Off((j-2)*nb, j*nb), czero, work.CMatrixOff(j*nb, *n, opts))
					}

					//                 Update with the previous column
					err = goblas.Zgemm(Trans, NoTrans, nb, (*n)-(j+1)*nb, j*nb, -cone, work.CMatrixOff(nb, *n, opts), a.Off(0, (j+1)*nb), cone, a.Off(j*nb, (j+1)*nb))
				}

				//              Copy panel to workspace to call ZGETRF
				for k = 1; k <= nb; k++ {
					goblas.Zcopy((*n)-(j+1)*nb, a.CVector(j*nb+k-1, (j+1)*nb, *lda), work.Off(1+(k-1)*(*n)-1, 1))
				}

				//              Factorize panel
				Zgetrf(toPtr((*n)-(j+1)*nb), &nb, work.CMatrix(*n, opts), n, toSlice(ipiv, (j+1)*nb), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Copy panel back
				for k = 1; k <= nb; k++ {
					goblas.Zcopy((*n)-(j+1)*nb, work.Off(1+(k-1)*(*n)-1, 1), a.CVector(j*nb+k-1, (j+1)*nb, *lda))
				}

				//              Compute T(J+1, J), zero out for GEMM update
				kb = min(nb, (*n)-(j+1)*nb)
				Zlaset('F', &kb, &nb, &czero, &czero, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Zlacpy('U', &kb, &nb, work.CMatrix(*n, opts), n, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					err = goblas.Ztrsm(Right, Upper, NoTrans, Unit, kb, nb, cone, a.Off((j-1)*nb, j*nb), tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts))
				}

				//              Copy T(J,J+1) into T(J+1, J), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Zlaset('L', &kb, &nb, &czero, &cone, a.Off(j*nb, (j+1)*nb), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Zswap(k-1, a.CVector((j+1)*nb, i1-1, 1), a.CVector((j+1)*nb, i2-1, 1))
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Zswap(i2-i1-1, a.CVector(i1-1, i1, *lda), a.CVector(i1, i2-1, 1))
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Zswap((*n)-i2, a.CVector(i1-1, i2, *lda), a.CVector(i2-1, i2, *lda))
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Zswap(j*nb, a.CVector(0, i1-1, 1), a.CVector(0, i2-1, 1))
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
			kb = min(nb, (*n)-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                  H(I,J) = T(I,I)*L(J,I)' + T(I+1,I)'*L(J,I+1)'
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					err = goblas.Zgemm(NoTrans, Trans, nb, kb, jb, cone, tb.CMatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), a.Off(j*nb, (i-1)*nb), czero, work.CMatrixOff(i*nb, *n, opts))
				} else {
					//                 H(I,J) = T(I,I-1)*L(J,I-1)' + T(I,I)*L(J,I)' + T(I,I+1)*L(J,I+1)'
					if i == (j - 1) {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					err = goblas.Zgemm(NoTrans, Trans, nb, kb, jb, cone, tb.CMatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), a.Off(j*nb, (i-2)*nb), czero, work.CMatrixOff(i*nb, *n, opts))
				}
			}

			//           Compute T(J,J)
			Zlacpy('L', &kb, &kb, a.Off(j*nb, j*nb), lda, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = L(J,1:J)*H(1:J)
				err = goblas.Zgemm(NoTrans, NoTrans, kb, kb, (j-1)*nb, -cone, a.Off(j*nb, 0), work.CMatrixOff(nb, *n, opts), cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
				//              T(J,J) += L(J,J)*T(J,J-1)*L(J,J-1)'
				err = goblas.Zgemm(NoTrans, NoTrans, kb, nb, kb, cone, a.Off(j*nb, (j-1)*nb), tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), czero, work.CMatrix(*n, opts))
				err = goblas.Zgemm(NoTrans, Trans, kb, kb, nb, -cone, work.CMatrix(*n, opts), a.Off(j*nb, (j-2)*nb), cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
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
				err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, kb, kb, cone, a.Off(j*nb, (j-1)*nb), tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
				err = goblas.Ztrsm(Right, Lower, Trans, NonUnit, kb, kb, cone, a.Off(j*nb, (j-1)*nb), tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts))
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
						err = goblas.Zgemm(NoTrans, Trans, kb, kb, kb, cone, tb.CMatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), a.Off(j*nb, (j-1)*nb), czero, work.CMatrixOff(j*nb, *n, opts))
					} else {
						err = goblas.Zgemm(NoTrans, Trans, kb, kb, nb+kb, cone, tb.CMatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), a.Off(j*nb, (j-2)*nb), czero, work.CMatrixOff(j*nb, *n, opts))
					}

					//                 Update with the previous column
					err = goblas.Zgemm(NoTrans, NoTrans, (*n)-(j+1)*nb, nb, j*nb, -cone, a.Off((j+1)*nb, 0), work.CMatrixOff(nb, *n, opts), cone, a.Off((j+1)*nb, j*nb))
				}

				//              Factorize panel
				Zgetrf(toPtr((*n)-(j+1)*nb), &nb, a.Off((j+1)*nb, j*nb), lda, toSlice(ipiv, (j+1)*nb), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Compute T(J+1, J), zero out for GEMM update
				kb = min(nb, (*n)-(j+1)*nb)
				Zlaset('F', &kb, &nb, &czero, &czero, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Zlacpy('U', &kb, &nb, a.Off((j+1)*nb, j*nb), lda, tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					err = goblas.Ztrsm(Right, Lower, Trans, Unit, kb, nb, cone, a.Off(j*nb, (j-1)*nb), tb.CMatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts))
				}

				//              Copy T(J+1,J) into T(J, J+1), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Zlaset('U', &kb, &nb, &czero, &cone, a.Off((j+1)*nb, j*nb), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Zswap(k-1, a.CVector(i1-1, (j+1)*nb, *lda), a.CVector(i2-1, (j+1)*nb, *lda))
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Zswap(i2-i1-1, a.CVector(i1, i1-1, 1), a.CVector(i2-1, i1, *lda))
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Zswap((*n)-i2, a.CVector(i2, i1-1, 1), a.CVector(i2, i2-1, 1))
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Zswap(j*nb, a.CVector(i1-1, 0, *lda), a.CVector(i2-1, 0, *lda))
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
