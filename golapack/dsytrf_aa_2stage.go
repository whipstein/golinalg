package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrfAa2stage computes the factorization of a real symmetric matrix A
// using the Aasen's algorithm.  The form of the factorization is
//
//    A = U**T*T*U  or  A = L*T*L**T
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is a symmetric band matrix with the
// bandwidth of NB (NB is internally selected and stored in TB( 1 ), and T is
// LU factorized with partial pivoting).
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func DsytrfAa2stage(uplo byte, n *int, a *mat.Matrix, lda *int, tb *mat.Vector, ltb *int, ipiv, ipiv2 *[]int, work *mat.Vector, lwork, info *int) {
	var tquery, upper, wquery bool
	var one, piv, zero float64
	var i, i1, i2, iinfo, j, jb, k, kb, ldtb, nb, nt, td int

	zero = 0.0
	one = 1.0

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
		gltest.Xerbla([]byte("DSYTRF_AA_2STAGE"), -(*info))
		return
	}

	//     Answer the query
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRF_AA_2STAGE"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if (*info) == 0 {
		if tquery {
			tb.Set(0, float64((3*nb+1)*(*n)))
		}
		if wquery {
			work.Set(0, float64((*n)*nb))
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
	tb.Set(0, float64(nb))

	if upper {
		//        .....................................................
		//        Factorize A as U**T*D*U using the upper triangle of A
		//        .....................................................
		for j = 0; j <= nt-1; j++ {
			//           Generate Jth column of W and H
			kb = minint(nb, (*n)-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                 H(I,J) = T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					goblas.Dgemm(NoTrans, NoTrans, &nb, &kb, &jb, &one, tb.MatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((i-1)*nb+1-1, j*nb+1-1), lda, &zero, work.MatrixOff(i*nb+1-1, *n, opts), n)
				} else {
					//                 H(I,J) = T(I,I-1)*U(I-1,J) + T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == j-1 {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					goblas.Dgemm(NoTrans, NoTrans, &nb, &kb, &jb, &one, tb.MatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((i-2)*nb+1-1, j*nb+1-1), lda, &zero, work.MatrixOff(i*nb+1-1, *n, opts), n)
				}
			}

			//           Compute T(J,J)
			Dlacpy('U', &kb, &kb, a.Off(j*nb+1-1, j*nb+1-1), lda, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = U(1:J,J)'*H(1:J)
				goblas.Dgemm(Trans, NoTrans, &kb, &kb, toPtr((j-1)*nb), toPtrf64(-one), a.Off(0, j*nb+1-1), lda, work.MatrixOff(nb+1-1, *n, opts), n, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				//              T(J,J) += U(J,J)'*T(J,J-1)*U(J-1,J)
				goblas.Dgemm(Trans, NoTrans, &kb, &nb, &kb, &one, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.MatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), &zero, work.Matrix(*n, opts), n)
				goblas.Dgemm(NoTrans, NoTrans, &kb, &kb, &nb, toPtrf64(-one), work.Matrix(*n, opts), n, a.Off((j-2)*nb+1-1, j*nb+1-1), lda, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}
			if j > 0 {
				Dsygst(func() *int { y := 1; return &y }(), 'U', &kb, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((j-1)*nb+1-1, j*nb+1-1), lda, &iinfo)
			}

			//           Expand T(J,J) into full format
			for i = 1; i <= kb; i++ {
				for k = i + 1; k <= kb; k++ {
					tb.Set(td+(k-i)+1+(j*nb+i-1)*ldtb-1, tb.Get(td-(k-(i+1))+(j*nb+k-1)*ldtb-1))
				}
			}

			if j < nt-1 {
				if j > 0 {
					//                 Compute H(J,J)
					if j == 1 {
						goblas.Dgemm(NoTrans, NoTrans, &kb, &kb, &kb, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((j-1)*nb+1-1, j*nb+1-1), lda, &zero, work.MatrixOff(j*nb+1-1, *n, opts), n)
					} else {
						goblas.Dgemm(NoTrans, NoTrans, &kb, &kb, toPtr(nb+kb), &one, tb.MatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off((j-2)*nb+1-1, j*nb+1-1), lda, &zero, work.MatrixOff(j*nb+1-1, *n, opts), n)
					}

					//                 Update with the previous column
					goblas.Dgemm(Trans, NoTrans, &nb, toPtr((*n)-(j+1)*nb), toPtr(j*nb), toPtrf64(-one), work.MatrixOff(nb+1-1, *n, opts), n, a.Off(0, (j+1)*nb+1-1), lda, &one, a.Off(j*nb+1-1, (j+1)*nb+1-1), lda)
				}

				//              Copy panel to workspace to call DGETRF
				for k = 1; k <= nb; k++ {
					goblas.Dcopy(toPtr((*n)-(j+1)*nb), a.Vector(j*nb+k-1, (j+1)*nb+1-1), lda, work.Off(1+(k-1)*(*n)-1), func() *int { y := 1; return &y }())
				}

				//              Factorize panel
				Dgetrf(toPtr((*n)-(j+1)*nb), &nb, work.Matrix(*n, opts), n, toSlice(ipiv, (j+1)*nb+1-1), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Copy panel back
				for k = 1; k <= nb; k++ {
					goblas.Dcopy(toPtr((*n)-(j+1)*nb), work.Off(1+(k-1)*(*n)-1), func() *int { y := 1; return &y }(), a.Vector(j*nb+k-1, (j+1)*nb+1-1), lda)
				}

				//              Compute T(J+1, J), zero out for GEMM update
				kb = minint(nb, (*n)-(j+1)*nb)
				Dlaset('F', &kb, &nb, &zero, &zero, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Dlacpy('U', &kb, &nb, work.Matrix(*n, opts), n, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					goblas.Dtrsm(Right, Upper, NoTrans, Unit, &kb, &nb, &one, a.Off((j-1)*nb+1-1, j*nb+1-1), lda, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				}

				//              Copy T(J,J+1) into T(J+1, J), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Dlaset('L', &kb, &nb, &zero, &one, a.Off(j*nb+1-1, (j+1)*nb+1-1), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Dswap(toPtr(k-1), a.Vector((j+1)*nb+1-1, i1-1), func() *int { y := 1; return &y }(), a.Vector((j+1)*nb+1-1, i2-1), func() *int { y := 1; return &y }())
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Dswap(toPtr(i2-i1-1), a.Vector(i1-1, i1+1-1), lda, a.Vector(i1+1-1, i2-1), func() *int { y := 1; return &y }())
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Dswap(toPtr((*n)-i2), a.Vector(i1-1, i2+1-1), lda, a.Vector(i2-1, i2+1-1), lda)
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Dswap(toPtr(j*nb), a.Vector(0, i1-1), func() *int { y := 1; return &y }(), a.Vector(0, i2-1), func() *int { y := 1; return &y }())
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
					if i == j-1 {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					goblas.Dgemm(NoTrans, Trans, &nb, &kb, &jb, &one, tb.MatrixOff(td+1+(i*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (i-1)*nb+1-1), lda, &zero, work.MatrixOff(i*nb+1-1, *n, opts), n)
				} else {
					//                 H(I,J) = T(I,I-1)*L(J,I-1)' + T(I,I)*L(J,I)' + T(I,I+1)*L(J,I+1)'
					if i == j-1 {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					goblas.Dgemm(NoTrans, Trans, &nb, &kb, &jb, &one, tb.MatrixOff(td+nb+1+((i-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (i-2)*nb+1-1), lda, &zero, work.MatrixOff(i*nb+1-1, *n, opts), n)
				}
			}

			//           Compute T(J,J)
			Dlacpy('L', &kb, &kb, a.Off(j*nb+1-1, j*nb+1-1), lda, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			if j > 1 {
				//              T(J,J) = L(J,1:J)*H(1:J)
				goblas.Dgemm(NoTrans, NoTrans, &kb, &kb, toPtr((j-1)*nb), toPtrf64(-one), a.Off(j*nb+1-1, 0), lda, work.MatrixOff(nb+1-1, *n, opts), n, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				//              T(J,J) += L(J,J)*T(J,J-1)*L(J,J-1)'
				goblas.Dgemm(NoTrans, NoTrans, &kb, &nb, &kb, &one, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.MatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), &zero, work.Matrix(*n, opts), n)
				goblas.Dgemm(NoTrans, Trans, &kb, &kb, &nb, toPtrf64(-one), work.Matrix(*n, opts), n, a.Off(j*nb+1-1, (j-2)*nb+1-1), lda, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
			}
			if j > 0 {
				Dsygst(func() *int { y := 1; return &y }(), 'L', &kb, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, &iinfo)
			}

			//           Expand T(J,J) into full format
			for i = 1; i <= kb; i++ {
				for k = i + 1; k <= kb; k++ {
					tb.Set(td-(k-(i+1))+(j*nb+k-1)*ldtb-1, tb.Get(td+(k-i)+1+(j*nb+i-1)*ldtb-1))
				}
			}

			if j < nt-1 {
				if j > 0 {
					//                 Compute H(J,J)
					if j == 1 {
						goblas.Dgemm(NoTrans, Trans, &kb, &kb, &kb, &one, tb.MatrixOff(td+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, &zero, work.MatrixOff(j*nb+1-1, *n, opts), n)
					} else {
						goblas.Dgemm(NoTrans, Trans, &kb, &kb, toPtr(nb+kb), &one, tb.MatrixOff(td+nb+1+((j-1)*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1), a.Off(j*nb+1-1, (j-2)*nb+1-1), lda, &zero, work.MatrixOff(j*nb+1-1, *n, opts), n)
					}

					//                 Update with the previous column
					goblas.Dgemm(NoTrans, NoTrans, toPtr((*n)-(j+1)*nb), &nb, toPtr(j*nb), toPtrf64(-one), a.Off((j+1)*nb+1-1, 0), lda, work.MatrixOff(nb+1-1, *n, opts), n, &one, a.Off((j+1)*nb+1-1, j*nb+1-1), lda)
				}

				//              Factorize panel
				Dgetrf(toPtr((*n)-(j+1)*nb), &nb, a.Off((j+1)*nb+1-1, j*nb+1-1), lda, toSlice(ipiv, (j+1)*nb+1-1), &iinfo)
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Compute T(J+1, J), zero out for GEMM update
				kb = minint(nb, (*n)-(j+1)*nb)
				Dlaset('F', &kb, &nb, &zero, &zero, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				Dlacpy('U', &kb, &nb, a.Off((j+1)*nb+1-1, j*nb+1-1), lda, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				if j > 0 {
					goblas.Dtrsm(Right, Lower, Trans, Unit, &kb, &nb, &one, a.Off(j*nb+1-1, (j-1)*nb+1-1), lda, tb.MatrixOff(td+nb+1+(j*nb)*ldtb-1, ldtb-1, opts), toPtr(ldtb-1))
				}

				//              Copy T(J+1,J) into T(J, J+1), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Dlaset('U', &kb, &nb, &zero, &one, a.Off((j+1)*nb+1-1, j*nb+1-1), lda)

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						goblas.Dswap(toPtr(k-1), a.Vector(i1-1, (j+1)*nb+1-1), lda, a.Vector(i2-1, (j+1)*nb+1-1), lda)
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							goblas.Dswap(toPtr(i2-i1-1), a.Vector(i1+1-1, i1-1), func() *int { y := 1; return &y }(), a.Vector(i2-1, i1+1-1), lda)
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < (*n) {
							goblas.Dswap(toPtr((*n)-i2), a.Vector(i2+1-1, i1-1), func() *int { y := 1; return &y }(), a.Vector(i2+1-1, i2-1), func() *int { y := 1; return &y }())
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							goblas.Dswap(toPtr(j*nb), a.Vector(i1-1, 0), lda, a.Vector(i2-1, 0), lda)
						}
					}
				}

				//              Apply pivots to previous columns of L
				//
				//c               CALL DLASWP( J*NB, A( 1, 1 ), LDA,
				//c     $                     (J+1)*NB+1, (J+1)*NB+KB, IPIV, 1 )
			}
		}
	}

	//     Factor the band matrix
	Dgbtrf(n, n, &nb, &nb, tb.Matrix(ldtb, opts), &ldtb, ipiv2, info)
}
