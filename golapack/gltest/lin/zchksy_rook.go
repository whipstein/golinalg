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

// zchksyRook tests ZsytrfRook, -TRI_ROOK, -TRS_ROOK,
// and -CON_ROOK.
func zchksyRook(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var czero complex128
	var alpha, anorm, cndnum, _const, dtemp, eight, one, onehalf, rcond, rcondc, sevten, singMax, singMin, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

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

	//     Initialize constants and the random number seed.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Test path
	path := "Zsr"
	alasumStart(path)

	//     Path to generate matrices
	matpath := "Zsy"

	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrsy(path, t)
	}
	(*infot) = 0

	//     Set the minimum block size for which the block routine should
	//     be used, which will be later returned by ILAENV
	xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= nn; in++ {
		n = nval[in-1]
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
			if !dotype[imat-1] {
				goto label260
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label260
			}

			//           Do first for uplo='U', then for uplo='L'
			for _, uplo = range mat.IterMatUplo(false) {

				//              Begin generate test matrix A.
				if imat != ntypes {
					//                 Set up parameters with ZLATB4 for the matrix generator
					//                 based on the _type of matrix to be generated.
					_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(matpath, imat, n, n)

					//                 Generate a matrix with Zlatms.
					*srnamt = "Zlatms"
					if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.CMatrix(lda, opts), work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)

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
							if uplo == Upper {
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
							if uplo == Upper {
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
					zlatsy(uplo, n, a.CMatrix(lda, opts), &iseed)

				}

				//              End generate test matrix A.
				//
				//
				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= nnb; inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					lwork = max(2, nb) * lda
					*srnamt = "ZsytrfRook"
					info, err = golapack.ZsytrfRook(uplo, n, afac.CMatrix(lda, opts), &iwork, ainv, lwork)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					k = izero
					if k > 0 {
					label100:
						;
						if iwork[k-1] < 0 {
							if iwork[k-1] != -k {
								k = -iwork[k-1]
								goto label100
							}
						} else if iwork[k-1] != k {
							k = iwork[k-1]
							goto label100
						}
					}

					//                 Check error code from ZsytrfRook and handle error.
					if err != nil || info != k {
						t.Fail()
						nerrs = alaerh(path, "ZsytrfRook", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					*result.GetPtr(0) = zsyt01Rook(uplo, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, ainv.CMatrix(lda, opts), rwork)
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual,
					//                 if the factorization was competed without INFO > 0
					//                 (i.e. there is no zero rows and columns).
					//                 Do it only for the first block size.
					if inb == 1 && !trfcon {
						golapack.Zlacpy(uplo, n, n, afac.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
						*srnamt = "ZsytriRook"
						if info, err = golapack.ZsytriRook(uplo, n, ainv.CMatrix(lda, opts), &iwork, work); err != nil || info != 0 {
							t.Fail()
							nerrs = alaerh(path, "ZsytriRook", info, -1, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						rcondc, *result.GetPtr(1) = zsyt03(uplo, n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)
						nt = 2
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, nb=%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//+    TEST 3
					//                 Compute largest element in U or L
					result.Set(2, zero)
					dtemp = zero

					_const = ((math.Pow(alpha, 2) - one) / (math.Pow(alpha, 2) - onehalf)) / (one - alpha)

					if uplo == Upper {
						//                 Compute largest element in U
						k = n
					label120:
						;
						if k <= 1 {
							goto label130
						}

						if iwork[k-1] > int(zero) {
							//                       Get max absolute value from elements
							//                       in column k in in U
							dtemp = golapack.Zlange('M', k-1, 1, afac.Off((k-1)*lda).CMatrix(lda, opts), rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k-1 in U
							dtemp = golapack.Zlange('M', k-2, 2, afac.Off((k-2)*lda).CMatrix(lda, opts), rwork)
							k = k - 1

						}

						//                    DTEMP should be bounded by CONST
						dtemp = dtemp - _const + thresh
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

						if iwork[k-1] > int(zero) {
							//                       Get max absolute value from elements
							//                       in column k in in L
							dtemp = golapack.Zlange('M', n-k, 1, afac.Off((k-1)*lda+k).CMatrix(lda, opts), rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k+1 in L
							dtemp = golapack.Zlange('M', n-k-1, 2, afac.Off((k-1)*lda+k+2-1).CMatrix(lda, opts), rwork)
							k = k + 1

						}

						//                    DTEMP should be bounded by CONST
						dtemp = dtemp - _const + thresh
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

					if uplo == Upper {
						//                    Loop backward for uplo='U'
						k = n
					label160:
						;
						if k <= 1 {
							goto label170
						}

						if iwork[k-1] < int(zero) {
							//                       Get the two singular values
							//                       (real and non-negative) of a 2-by-2 block,
							//                       store them in RWORK array
							block.Set(0, 0, afac.Get((k-2)*lda+k-1-1))
							block.Set(0, 1, afac.Get((k-1)*lda+k-1-1))
							block.Set(1, 0, block.Get(0, 1))
							block.Set(1, 1, afac.Get((k-1)*lda+k-1))

							if info, err = golapack.Zgesvd('N', 'N', 2, 2, block, rwork, zdummy.CMatrix(1, opts), zdummy.CMatrix(1, opts), work, 6, rwork.Off(2)); err != nil {
								panic(err)
							}

							singMax = rwork.Get(0)
							singMin = rwork.Get(1)

							dtemp = singMax / singMin

							//                       DTEMP should be bounded by CONST
							dtemp = dtemp - _const + thresh
							if dtemp > result.Get(3) {
								result.Set(3, dtemp)
							}
							k = k - 1

						}

						k = k - 1

						goto label160
					label170:
					} else {
						//                    Loop forward for uplo='L'
						k = 1
					label180:
						;
						if k >= n {
							goto label190
						}

						if iwork[k-1] < int(zero) {
							//                       Get the two singular values
							//                       (real and non-negative) of a 2-by-2 block,
							//                       store them in RWORK array
							block.Set(0, 0, afac.Get((k-1)*lda+k-1))
							block.Set(1, 0, afac.Get((k-1)*lda+k))
							block.Set(0, 1, block.Get(1, 0))
							block.Set(1, 1, afac.Get(k*lda+k))

							if info, err = golapack.Zgesvd('N', 'N', 2, 2, block, rwork, zdummy.CMatrix(1, opts), zdummy.CMatrix(1, opts), work, 6, rwork.Off(2)); err != nil {
								panic(err)
							}

							singMax = rwork.Get(0)
							singMin = rwork.Get(1)

							dtemp = singMax / singMin

							//                       DTEMP should be bounded by CONST
							dtemp = dtemp - _const + thresh
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
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, nb=%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 2

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

					//                 Do for each value of nrhs in NSVAL.
					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]

						//+    TEST 5 ( Using TRS_ROOK)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "zlarhs"
						if err = zlarhs(matpath, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "ZsytrsRook"
						if err = golapack.ZsytrsRook(uplo, n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts)); err != nil {
							t.Fail()
							nerrs = alaerh(path, "ZsytrsRook", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))

						//                    Compute the residual for the solution
						*result.GetPtr(4) = zsyt02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

						//+    TEST 6
						//                 Check solution from generated exact solution.
						*result.GetPtr(5) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 5; k <= 6; k++ {
							if result.Get(k-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun += 2

						//                 End do for each value of nrhs in NSVAL.
					}

					//+    TEST 7
					//                 Get an estimate of RCOND = 1/CNDNUM.
				label230:
					;
					anorm = golapack.Zlansy('1', uplo, n, a.CMatrix(lda, opts), rwork)
					*srnamt = "ZsyconRook"
					if rcond, err = golapack.ZsyconRook(uplo, n, afac.CMatrix(lda, opts), &iwork, anorm, work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "ZsyconRook", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					//                 Compute the test ratio to compare values of RCOND
					result.Set(6, dget06(rcond, rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(6) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" uplo=%s, n=%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 7, result.Get(6))
						nfail++
					}
					nrun++
				label240:
				}

			label250:
			}
		label260:
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
