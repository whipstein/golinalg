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

// DchksyRook tests DSYTRF_ROOK, -TRI_ROOK, -TRS_ROOK,
// and -CON_ROOK.
func DchksyRook(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, uplo, xtype byte
	var alpha, anorm, cndnum, _const, dtemp, eight, one, rcond, rcondc, sevten, singMax, singMin, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt
	block := mf(2, 2, opts)
	ddummy := vf(1)

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0
	ntypes = 10
	// ntests = 7

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Test path
	path := []byte("DSR")

	//     Path to generate matrices
	matpath := []byte("DSY")

	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrsy(path, t)
	}
	(*infot) = 0

	//     Set the minimum block size for which the block routine should
	//     be used, which will be later returned by ILAENV
	Xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
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
				//              Begin generate the test matrix A.
				//
				//              Set up parameters with DLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				Dlatb4(matpath, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				//              Generate a matrix with DLATMS.
				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS and handle error.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)

					//                 Skip all tests for this generated matrix
					goto label250
				}

				//              For matrix types 3-6, zero one or more rows and
				//              columns of the matrix to test that INFO is returned
				//              correctly.
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
								a.Set(ioff+i-1, zero)
							}
							ioff = ioff + izero
							for i = izero; i <= n; i++ {
								a.Set(ioff-1, zero)
								ioff = ioff + lda
							}
						} else {
							ioff = izero
							for i = 1; i <= izero-1; i++ {
								a.Set(ioff-1, zero)
								ioff = ioff + lda
							}
							ioff = ioff - izero
							for i = izero; i <= n; i++ {
								a.Set(ioff+i-1, zero)
							}
						}
					} else {
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = minint(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i1 = maxint(j, izero)
								for i = i1; i <= n; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						}
					}
				} else {
					izero = 0
				}

				//              End generate the test matrix A.
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
					golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					lwork = maxint(2, nb) * lda
					*srnamt = "DSYTRF_ROOK"
					golapack.DsytrfRook(uplo, &n, afac.Matrix(lda, opts), &lda, iwork, ainv.Matrix(lda, opts), &lwork, &info)

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

					//                 Check error code from DSYTRF_ROOK and handle error.
					if info != k {
						Alaerh(path, []byte("DSYTRF_ROOK"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					Dsyt01Rook(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual,
					//                 if the factorization was competed without INFO > 0
					//                 (i.e. there is no zero rows and columns).
					//                 Do it only for the first block size.
					if inb == 1 && !trfcon {
						golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
						*srnamt = "DSYTRI_ROOK"
						golapack.DsytriRook(uplo, &n, ainv.Matrix(lda, opts), &lda, iwork, work, &info)

						//                    Check error code from DSYTRI_ROOK and handle error.
						if info != 0 {
							Alaerh(path, []byte("DSYTRI_ROOK"), &info, toPtr(-1), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						Dpot03(uplo, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))
						nt = 2
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
							fmt.Printf(" UPLO = '%c', N =%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + nt

					//+    TEST 3
					//                 Compute largest element in U or L
					result.Set(2, zero)
					dtemp = zero

					_const = one / (one - alpha)

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
							dtemp = golapack.Dlange('M', toPtr(k-1), func() *int { y := 1; return &y }(), afac.MatrixOff(0+(k-1)*lda, lda, opts), &lda, rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k-1 in U
							dtemp = golapack.Dlange('M', toPtr(k-2), func() *int { y := 2; return &y }(), afac.MatrixOff(0+(k-2)*lda, lda, opts), &lda, rwork)
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
							dtemp = golapack.Dlange('M', toPtr(n-k), func() *int { y := 1; return &y }(), afac.MatrixOff(k+1-1+(k-1)*lda, lda, opts), &lda, rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k+1 in L
							dtemp = golapack.Dlange('M', toPtr(n-k-1), func() *int { y := 2; return &y }(), afac.MatrixOff(k+2-1+(k-1)*lda, lda, opts), &lda, rwork)
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

					_const = (one + alpha) / (one - alpha)
					golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)

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
							block.Set(0, 0, afac.Get(k-1-1+(k-2)*lda))
							block.Set(0, 1, afac.Get(k-1-1+(k-1)*lda))
							block.Set(1, 0, block.Get(0, 1))
							block.Set(1, 1, afac.Get(k-1+(k-1)*lda))

							golapack.Dgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), block, func() *int { y := 2; return &y }(), rwork, ddummy.Matrix(1, opts), func() *int { y := 1; return &y }(), ddummy.Matrix(1, opts), func() *int { y := 1; return &y }(), work, func() *int { y := 10; return &y }(), &info)

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
							block.Set(0, 0, afac.Get(k-1+(k-1)*lda))
							block.Set(1, 0, afac.Get(k+1-1+(k-1)*lda))
							block.Set(0, 1, block.Get(1, 0))
							block.Set(1, 1, afac.Get(k+1-1+(k)*lda))

							golapack.Dgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), block, func() *int { y := 2; return &y }(), rwork, ddummy.Matrix(1, opts), func() *int { y := 1; return &y }(), ddummy.Matrix(1, opts), func() *int { y := 1; return &y }(), work, func() *int { y := 10; return &y }(), &info)

							singMax = rwork.Get(0)
							singMin = rwork.Get(1)

							dtemp = singMax / singMin

							//                       DTEMP should be bounded by CONST
							dtemp = dtemp - _const + (*thresh)
							if dtemp > result.Get((3)) {
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
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
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
						*srnamt = "DLARHS"
						Dlarhs(matpath, &xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						*srnamt = "DSYTRS_ROOK"
						golapack.DsytrsRook(uplo, &n, &nrhs, afac.Matrix(lda, opts), &lda, iwork, x.Matrix(lda, opts), &lda, &info)

						//                    Check error code from DSYTRS_ROOK and handle error.
						if info != 0 {
							Alaerh(path, []byte("DSYTRS_ROOK"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)

						//                    Compute the residual for the solution
						Dpot02(uplo, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(4))

						//+    TEST 6
						//                 Check solution from generated exact solution.
						Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(5))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 5; k <= 6; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								t.Fail()
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
					anorm = golapack.Dlansy('1', uplo, &n, a.Matrix(lda, opts), &lda, rwork)
					*srnamt = "DSYCON_ROOK"
					golapack.DsyconRook(uplo, &n, afac.Matrix(lda, opts), &lda, iwork, &anorm, &rcond, work, toSlice(iwork, n+1-1), &info)

					//                 Check error code from DSYCON_ROOK and handle error.
					if info != 0 {
						Alaerh(path, []byte("DSYCON_ROOK"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					//                 Compute the test ratio to compare to values of RCOND
					result.Set(6, Dget06(&rcond, &rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(6) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
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

	//     Verify number of tests match original.
	tgtRuns := 1618
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
