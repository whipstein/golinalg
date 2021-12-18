package golapack

import (
	"fmt"

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
func DsytrfAa2stage(uplo mat.MatUplo, n int, a *mat.Matrix, tb *mat.Vector, ltb int, ipiv, ipiv2 *[]int, work *mat.Vector, lwork int) (info int, err error) {
	var tquery, upper, wquery bool
	var one, piv, zero float64
	var i, i1, i2, j, jb, k, kb, ldtb, nb, nt, td int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	wquery = (lwork == -1)
	tquery = (ltb == -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if ltb < 4*n && !tquery {
		err = fmt.Errorf("ltb < 4*n && !tquery: ltb=%v, n=%v, tquery=%v", ltb, n, tquery)
	} else if lwork < n && !wquery {
		err = fmt.Errorf("lwork < n && !wquery: lwork=%v, n=%v, wquery=%v", lwork, n, wquery)
	}

	if err != nil {
		gltest.Xerbla2("DsytrfAa2stage", err)
		return
	}

	//     Answer the query
	nb = Ilaenv(1, "DsytrfAa2stage", []byte{uplo.Byte()}, n, -1, -1, -1)
	if info == 0 {
		if tquery {
			tb.Set(0, float64((3*nb+1)*n))
		}
		if wquery {
			work.Set(0, float64(n*nb))
		}
	}
	if tquery || wquery {
		return
	}

	//     Quick return
	if n == 0 {
		return
	}

	//     Determine the number of the block size
	ldtb = ltb / n
	if ldtb < 3*nb+1 {
		nb = (ldtb - 1) / 3
	}
	if lwork < nb*n {
		nb = lwork / n
	}

	//     Determine the number of the block columns
	nt = (n + nb - 1) / nb
	td = 2 * nb
	kb = min(nb, n)

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
			kb = min(nb, n-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                 H(I,J) = T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == (j - 1) {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					if err = work.Off(i*nb).Matrix(n, opts).Gemm(NoTrans, NoTrans, nb, kb, jb, one, tb.Off(td+1+(i*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off((i-1)*nb, j*nb), zero); err != nil {
						panic(err)
					}
				} else {
					//                 H(I,J) = T(I,I-1)*U(I-1,J) + T(I,I)*U(I,J) + T(I,I+1)*U(I+1,J)
					if i == j-1 {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					if err = work.Off(i*nb).Matrix(n, opts).Gemm(NoTrans, NoTrans, nb, kb, jb, one, tb.Off(td+nb+1+((i-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off((i-2)*nb, j*nb), zero); err != nil {
						panic(err)
					}
				}
			}

			//           Compute T(J,J)
			Dlacpy(Upper, kb, kb, a.Off(j*nb, j*nb), tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
			if j > 1 {
				//              T(J,J) = U(1:J,J)'*H(1:J)
				if err = tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Gemm(Trans, NoTrans, kb, kb, (j-1)*nb, -one, a.Off(0, j*nb), work.Off(nb).Matrix(n, opts), one); err != nil {
					panic(err)
				}
				//              T(J,J) += U(J,J)'*T(J,J-1)*U(J-1,J)
				if err = work.Matrix(n, opts).Gemm(Trans, NoTrans, kb, nb, kb, one, a.Off((j-1)*nb, j*nb), tb.Off(td+nb+1+((j-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), zero); err != nil {
					panic(err)
				}
				if err = tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Gemm(NoTrans, NoTrans, kb, kb, nb, -one, work.Matrix(n, opts), a.Off((j-2)*nb, j*nb), one); err != nil {
					panic(err)
				}
			}
			if j > 0 {
				if err = Dsygst(1, Upper, kb, tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off((j-1)*nb, j*nb)); err != nil {
					panic(err)
				}
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
						if err = work.Off(j*nb).Matrix(n, opts).Gemm(NoTrans, NoTrans, kb, kb, kb, one, tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off((j-1)*nb, j*nb), zero); err != nil {
							panic(err)
						}
					} else {
						if err = work.Off(j*nb).Matrix(n, opts).Gemm(NoTrans, NoTrans, kb, kb, nb+kb, one, tb.Off(td+nb+1+((j-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off((j-2)*nb, j*nb), zero); err != nil {
							panic(err)
						}
					}

					//                 Update with the previous column
					if err = a.Off(j*nb, (j+1)*nb).Gemm(Trans, NoTrans, nb, n-(j+1)*nb, j*nb, -one, work.Off(nb).Matrix(n, opts), a.Off(0, (j+1)*nb), one); err != nil {
						panic(err)
					}
				}

				//              Copy panel to workspace to call DGETRF
				for k = 1; k <= nb; k++ {
					work.Off(1+(k-1)*n-1).Copy(n-(j+1)*nb, a.Off(j*nb+k-1, (j+1)*nb).Vector(), a.Rows, 1)
				}

				//              Factorize panel
				if _, err = Dgetrf(n-(j+1)*nb, nb, work.Matrix(n, opts), toSlice(ipiv, (j+1)*nb)); err != nil {
					panic(err)
				}
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Copy panel back
				for k = 1; k <= nb; k++ {
					a.Off(j*nb+k-1, (j+1)*nb).Vector().Copy(n-(j+1)*nb, work.Off(1+(k-1)*n-1), 1, a.Rows)
				}

				//              Compute T(J+1, J), zero out for GEMM update
				kb = min(nb, n-(j+1)*nb)
				Dlaset(Full, kb, nb, zero, zero, tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
				Dlacpy(Upper, kb, nb, work.Matrix(n, opts), tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
				if j > 0 {
					err = tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Trsm(Right, Upper, NoTrans, Unit, kb, nb, one, a.Off((j-1)*nb, j*nb))
				}

				//              Copy T(J,J+1) into T(J+1, J), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Dlaset(Lower, kb, nb, zero, one, a.Off(j*nb, (j+1)*nb))

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						a.Off((j+1)*nb, i2-1).Vector().Swap(k-1, a.Off((j+1)*nb, i1-1).Vector(), 1, 1)
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							a.Off(i1, i2-1).Vector().Swap(i2-i1-1, a.Off(i1-1, i1).Vector(), a.Rows, 1)
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < n {
							a.Off(i2-1, i2).Vector().Swap(n-i2, a.Off(i1-1, i2).Vector(), a.Rows, a.Rows)
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							a.Off(0, i2-1).Vector().Swap(j*nb, a.Off(0, i1-1).Vector(), 1, 1)
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
			kb = min(nb, n-j*nb)
			for i = 1; i <= j-1; i++ {
				if i == 1 {
					//                  H(I,J) = T(I,I)*L(J,I)' + T(I+1,I)'*L(J,I+1)'
					if i == j-1 {
						jb = nb + kb
					} else {
						jb = 2 * nb
					}
					if err = work.Off(i*nb).Matrix(n, opts).Gemm(NoTrans, Trans, nb, kb, jb, one, tb.Off(td+1+(i*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off(j*nb, (i-1)*nb), zero); err != nil {
						panic(err)
					}
				} else {
					//                 H(I,J) = T(I,I-1)*L(J,I-1)' + T(I,I)*L(J,I)' + T(I,I+1)*L(J,I+1)'
					if i == j-1 {
						jb = 2*nb + kb
					} else {
						jb = 3 * nb
					}
					if err = work.Off(i*nb).Matrix(n, opts).Gemm(NoTrans, Trans, nb, kb, jb, one, tb.Off(td+nb+1+((i-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off(j*nb, (i-2)*nb), zero); err != nil {
						panic(err)
					}
				}
			}

			//           Compute T(J,J)
			Dlacpy(Lower, kb, kb, a.Off(j*nb, j*nb), tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
			if j > 1 {
				//              T(J,J) = L(J,1:J)*H(1:J)
				if err = tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Gemm(NoTrans, NoTrans, kb, kb, (j-1)*nb, -one, a.Off(j*nb, 0), work.Off(nb).Matrix(n, opts), one); err != nil {
					panic(err)
				}
				//              T(J,J) += L(J,J)*T(J,J-1)*L(J,J-1)'
				if err = work.Matrix(n, opts).Gemm(NoTrans, NoTrans, kb, nb, kb, one, a.Off(j*nb, (j-1)*nb), tb.Off(td+nb+1+((j-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), zero); err != nil {
					panic(err)
				}
				if err = tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Gemm(NoTrans, Trans, kb, kb, nb, -one, work.Matrix(n, opts), a.Off(j*nb, (j-2)*nb), one); err != nil {
					panic(err)
				}
			}
			if j > 0 {
				if err = Dsygst(1, Lower, kb, tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off(j*nb, (j-1)*nb)); err != nil {
					panic(err)
				}
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
						if err = work.Off(j*nb).Matrix(n, opts).Gemm(NoTrans, Trans, kb, kb, kb, one, tb.Off(td+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off(j*nb, (j-1)*nb), zero); err != nil {
							panic(err)
						}
					} else {
						if err = work.Off(j*nb).Matrix(n, opts).Gemm(NoTrans, Trans, kb, kb, nb+kb, one, tb.Off(td+nb+1+((j-1)*nb)*ldtb-1).Matrix(ldtb-1, opts), a.Off(j*nb, (j-2)*nb), zero); err != nil {
							panic(err)
						}
					}

					//                 Update with the previous column
					if err = a.Off((j+1)*nb, j*nb).Gemm(NoTrans, NoTrans, n-(j+1)*nb, nb, j*nb, -one, a.Off((j+1)*nb, 0), work.Off(nb).Matrix(n, opts), one); err != nil {
						panic(err)
					}
				}

				//              Factorize panel
				if _, err = Dgetrf(n-(j+1)*nb, nb, a.Off((j+1)*nb, j*nb), toSlice(ipiv, (j+1)*nb)); err != nil {
					panic(err)
				}
				//c               IF (IINFO.NE.0 .AND. INFO.EQ.0) THEN
				//c                  INFO = IINFO+(J+1)*NB
				//c               END IF
				//
				//              Compute T(J+1, J), zero out for GEMM update
				kb = min(nb, n-(j+1)*nb)
				Dlaset(Full, kb, nb, zero, zero, tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
				Dlacpy(Upper, kb, nb, a.Off((j+1)*nb, j*nb), tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts))
				if j > 0 {
					err = tb.Off(td+nb+1+(j*nb)*ldtb-1).Matrix(ldtb-1, opts).Trsm(Right, Lower, Trans, Unit, kb, nb, one, a.Off(j*nb, (j-1)*nb))
				}

				//              Copy T(J+1,J) into T(J, J+1), both upper/lower for GEMM
				//              updates
				for k = 1; k <= nb; k++ {
					for i = 1; i <= kb; i++ {
						tb.Set(td-nb+k-i+1+(j*nb+nb+i-1)*ldtb-1, tb.Get(td+nb+i-k+1+(j*nb+k-1)*ldtb-1))
					}
				}
				Dlaset(Upper, kb, nb, zero, one, a.Off((j+1)*nb, j*nb))

				//              Apply pivots to trailing submatrix of A
				for k = 1; k <= kb; k++ {
					//                 > Adjust ipiv
					(*ipiv)[(j+1)*nb+k-1] = (*ipiv)[(j+1)*nb+k-1] + (j+1)*nb
					//
					i1 = (j+1)*nb + k
					i2 = (*ipiv)[(j+1)*nb+k-1]
					if i1 != i2 {
						//                    > Apply pivots to previous columns of L
						a.Off(i2-1, (j+1)*nb).Vector().Swap(k-1, a.Off(i1-1, (j+1)*nb).Vector(), a.Rows, a.Rows)
						//                    > Swap A(I1+1:M, I1) with A(I2, I1+1:M)
						if i2 > (i1 + 1) {
							a.Off(i2-1, i1).Vector().Swap(i2-i1-1, a.Off(i1, i1-1).Vector(), 1, a.Rows)
						}
						//                    > Swap A(I2+1:M, I1) with A(I2+1:M, I2)
						if i2 < n {
							a.Off(i2, i2-1).Vector().Swap(n-i2, a.Off(i2, i1-1).Vector(), 1, 1)
						}
						//                    > Swap A(I1, I1) with A(I2, I2)
						piv = a.Get(i1-1, i1-1)
						a.Set(i1-1, i1-1, a.Get(i2-1, i2-1))
						a.Set(i2-1, i2-1, piv)
						//                    > Apply pivots to previous columns of L
						if j > 0 {
							a.Off(i2-1, 0).Vector().Swap(j*nb, a.Off(i1-1, 0).Vector(), a.Rows, a.Rows)
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
	if info, err = Dgbtrf(n, n, nb, nb, tb.Matrix(ldtb, opts), ipiv2); err != nil {
		panic(err)
	}

	return
}
