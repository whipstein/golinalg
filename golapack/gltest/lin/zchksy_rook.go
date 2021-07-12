package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchksyrook tests ZSYTRF_ROOK, -TRI_ROOK, -TRS_ROOK,
// and -CON_ROOK.
func Zchksyrook(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, uplo, xtype byte
	var czero complex128
	var alpha, anorm, cndnum, _const, dtemp, eight, one, onehalf, rcond, rcondc, sevten, singMax, singMin, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	zdummy := cvf(1)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	block := cmf(2, 2, opts)
	zero = 0.0
	one = 1.0
	onehalf = 0.5
	eight = 8.0
	sevten = 17.0
	czero = (0.0 + 0.0*1i)
	ntypes = 11
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Test path
	path := []byte("ZSR")

	//     Path to generate matrices
	matpath := []byte("ZSY")

	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrsy(path, t)
	}
	(*infot) = 0

	//     Set the minimum block size for which the block routine should
	//     be used, which will be later returned by ILAENV
	Xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = max(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		izero = 0

		//        Do for each value of matrix _type IMAT
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label260
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label260
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]

				//              Begin generate test matrix A.
				if imat != ntypes {
					//                 Set up parameters with ZLATB4 for the matrix generator
					//                 based on the _type of matrix to be generated.
					Zlatb4(matpath, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

					//                 Generate a matrix with ZLATMS.
					*srnamt = "ZLATMS"
					matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.CMatrix(lda, opts), &lda, work, &info)

					//                 Check error code from ZLATMS and handle error.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)

						//                    Skip all tests for this generated matrix
						goto label250
					}

					//                 For matrix types 3-6, zero one or more rows and
					//                 columns of the matrix to test that INFO is returned
					//                 correctly.
					if zerot {
						if imat == 3 {
							izero = 1
						} else if imat == 4 {
							izero = n
						} else {
							izero = n/2 + 1
						}

						if imat < 6 {
							//                    Set row and column IZERO to zero.
							if iuplo == 1 {
								ioff = (izero - 1) * lda
								for i = 1; i <= izero-1; i++ {
									a.Set(ioff+i-1, czero)
								}
								ioff = ioff + izero
								for i = izero; i <= n; i++ {
									a.Set(ioff-1, czero)
									ioff = ioff + lda
								}
							} else {
								ioff = izero
								for i = 1; i <= izero-1; i++ {
									a.Set(ioff-1, czero)
									ioff = ioff + lda
								}
								ioff = ioff - izero
								for i = izero; i <= n; i++ {
									a.Set(ioff+i-1, czero)
								}
							}
						} else {
							if iuplo == 1 {
								//                          Set the first IZERO rows and columns to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i2 = min(j, izero)
									for i = 1; i <= i2; i++ {
										a.Set(ioff+i-1, czero)
									}
									ioff = ioff + lda
								}
							} else {
								//                          Set the last IZERO rows and columns to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i1 = max(j, izero)
									for i = i1; i <= n; i++ {
										a.Set(ioff+i-1, czero)
									}
									ioff = ioff + lda
								}
							}
						}
					} else {
						izero = 0
					}

				} else {
					//                 For matrix kind IMAT = 11, generate special block
					//                 diagonal matrix to test alternate code
					//                 for the 2 x 2 blocks.
					Zlatsy(uplo, &n, a.CMatrix(lda, opts), &lda, &iseed)

				}

				//              End generate test matrix A.
				//
				//
				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= (*nnb); inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					lwork = max(2, nb) * lda
					*srnamt = "ZSYTRF_ROOK"
					golapack.Zsytrfrook(uplo, &n, afac.CMatrix(lda, opts), &lda, iwork, ainv, &lwork, &info)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					k = izero
					if k > 0 {
					label100:
						;
						if (*iwork)[k-1] < 0 {
							if (*iwork)[k-1] != -k {
								k = -(*iwork)[k-1]
								goto label100
							}
						} else if (*iwork)[k-1] != k {
							k = (*iwork)[k-1]
							goto label100
						}
					}

					//                 Check error code from ZSYTRF_ROOK and handle error.
					if info != k {
						t.Fail()
						Alaerh(path, []byte("ZSYTRF_ROOK"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					Zsyt01rook(uplo, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, ainv.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual,
					//                 if the factorization was competed without INFO > 0
					//                 (i.e. there is no zero rows and columns).
					//                 Do it only for the first block size.
					if inb == 1 && !trfcon {
						golapack.Zlacpy(uplo, &n, &n, afac.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
						*srnamt = "ZSYTRI_ROOK"
						golapack.Zsytrirook(uplo, &n, ainv.CMatrix(lda, opts), &lda, iwork, work, &info)

						//                    Check error code from ZSYTRI_ROOK and handle error.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZSYTRI_ROOK"), &info, toPtr(-1), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						Zsyt03(uplo, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))
						nt = 2
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO = '%c', N =%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + nt

					//+    TEST 3
					//                 Compute largest element in U or L
					result.Set(2, zero)
					dtemp = zero

					_const = ((math.Pow(alpha, 2) - one) / (math.Pow(alpha, 2) - onehalf)) / (one - alpha)

					if iuplo == 1 {
						//                 Compute largest element in U
						k = n
					label120:
						;
						if k <= 1 {
							goto label130
						}

						if (*iwork)[k-1] > int(zero) {
							//                       Get max absolute value from elements
							//                       in column k in in U
							dtemp = golapack.Zlange('M', toPtr(k-1), func() *int { y := 1; return &y }(), afac.CMatrixOff((k-1)*lda, lda, opts), &lda, rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k-1 in U
							dtemp = golapack.Zlange('M', toPtr(k-2), func() *int { y := 2; return &y }(), afac.CMatrixOff((k-2)*lda, lda, opts), &lda, rwork)
							k = k - 1

						}

						//                    DTEMP should be bounded by CONST
						dtemp = dtemp - _const + (*thresh)
						if dtemp > result.Get(2) {
							result.Set(2, dtemp)
						}

						k = k - 1

						goto label120
					label130:
					} else {
						//                 Compute largest element in L
						k = 1
					label140:
						;
						if k >= n {
							goto label150
						}

						if (*iwork)[k-1] > int(zero) {
							//                       Get max absolute value from elements
							//                       in column k in in L
							dtemp = golapack.Zlange('M', toPtr(n-k), func() *int { y := 1; return &y }(), afac.CMatrixOff((k-1)*lda+k, lda, opts), &lda, rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k+1 in L
							dtemp = golapack.Zlange('M', toPtr(n-k-1), func() *int { y := 2; return &y }(), afac.CMatrixOff((k-1)*lda+k+2-1, lda, opts), &lda, rwork)
							k = k + 1

						}

						//                    DTEMP should be bounded by CONST
						dtemp = dtemp - _const + (*thresh)
						if dtemp > result.Get(2) {
							result.Set(2, dtemp)
						}

						k = k + 1

						goto label140
					label150:
					}

					//+    TEST 4
					//                 Compute largest 2-Norm (condition number)
					//                 of 2-by-2 diag blocks
					result.Set(3, zero)
					dtemp = zero

					_const = ((math.Pow(alpha, 2) - one) / (math.Pow(alpha, 2) - onehalf)) * ((one + alpha) / (one - alpha))

					if iuplo == 1 {
						//                    Loop backward for UPLO = 'U'
						k = n
					label160:
						;
						if k <= 1 {
							goto label170
						}

						if (*iwork)[k-1] < int(zero) {
							//                       Get the two singular values
							//                       (real and non-negative) of a 2-by-2 block,
							//                       store them in RWORK array
							block.Set(0, 0, afac.Get((k-2)*lda+k-1-1))
							block.Set(0, 1, afac.Get((k-1)*lda+k-1-1))
							block.Set(1, 0, block.Get(0, 1))
							block.Set(1, 1, afac.Get((k-1)*lda+k-1))

							golapack.Zgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), block, func() *int { y := 2; return &y }(), rwork, zdummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), zdummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), work, func() *int { y := 6; return &y }(), rwork.Off(2), &info)

							singMax = rwork.Get(0)
							singMin = rwork.Get(1)

							dtemp = singMax / singMin

							//                       DTEMP should be bounded by CONST
							dtemp = dtemp - _const + (*thresh)
							if dtemp > result.Get(3) {
								result.Set(3, dtemp)
							}
							k = k - 1

						}

						k = k - 1

						goto label160
					label170:
					} else {
						//                    Loop forward for UPLO = 'L'
						k = 1
					label180:
						;
						if k >= n {
							goto label190
						}

						if (*iwork)[k-1] < int(zero) {
							//                       Get the two singular values
							//                       (real and non-negative) of a 2-by-2 block,
							//                       store them in RWORK array
							block.Set(0, 0, afac.Get((k-1)*lda+k-1))
							block.Set(1, 0, afac.Get((k-1)*lda+k))
							block.Set(0, 1, block.Get(1, 0))
							block.Set(1, 1, afac.Get(k*lda+k))

							golapack.Zgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), block, func() *int { y := 2; return &y }(), rwork, zdummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), zdummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), work, func() *int { y := 6; return &y }(), rwork.Off(2), &info)

							singMax = rwork.Get(0)
							singMin = rwork.Get(1)

							dtemp = singMax / singMin

							//                       DTEMP should be bounded by CONST
							dtemp = dtemp - _const + (*thresh)
							if dtemp > result.Get(3) {
								result.Set(3, dtemp)
							}
							k = k + 1

						}

						k = k + 1

						goto label180
					label190:
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 3; k <= 4; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO = '%c', N =%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 2

					//                 Skip the other tests if this is not the first block
					//                 size.
					if inb > 1 {
						goto label240
					}

					//                 Do only the condition estimate if INFO is not 0.
					if trfcon {
						rcondc = zero
						goto label230
					}

					//                 Do for each value of NRHS in NSVAL.
					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]

						//+    TEST 5 ( Using TRS_ROOK)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of NRHS random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "ZLARHS"
						Zlarhs(matpath, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZSYTRS_ROOK"
						golapack.Zsytrsrook(uplo, &n, &nrhs, afac.CMatrix(lda, opts), &lda, iwork, x.CMatrix(lda, opts), &lda, &info)

						//                    Check error code from ZSYTRS_ROOK and handle error.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZSYTRS_ROOK"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)

						//                    Compute the residual for the solution
						Zsyt02(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(4))

						//+    TEST 6
						//                 Check solution from generated exact solution.
						Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(5))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 5; k <= 6; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + 2

						//                 End do for each value of NRHS in NSVAL.
					}

					//+    TEST 7
					//                 Get an estimate of RCOND = 1/CNDNUM.
				label230:
					;
					anorm = golapack.Zlansy('1', uplo, &n, a.CMatrix(lda, opts), &lda, rwork)
					*srnamt = "ZSYCON_ROOK"
					golapack.Zsyconrook(uplo, &n, afac.CMatrix(lda, opts), &lda, iwork, &anorm, &rcond, work, &info)

					//                 Check error code from ZSYCON_ROOK and handle error.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZSYCON_ROOK"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					//                 Compute the test ratio to compare values of RCOND
					result.Set(6, Dget06(&rcond, &rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(6) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" UPLO = '%c', N =%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 7, result.Get(6))
						nfail = nfail + 1
					}
					nrun = nrun + 1
				label240:
				}

			label250:
			}
		label260:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
