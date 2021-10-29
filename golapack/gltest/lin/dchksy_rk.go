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

// dchksyRk tests DSYTRF_RK, -TRI_3, -TRS_3, and -CON_3.
func dchksyRk(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, e, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var alpha, anorm, cndnum, _const, dtemp, eight, one, rcond, rcondc, sevten, singMax, singMin, zero float64
	var _result *float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt
	block := mf(2, 2, opts)
	ddummy := mf(1, 1, opts)

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0
	ntypes = 10

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Test path
	path := "Dsk"

	//     Path to generate matrices
	matpath := "Dsy"

	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrsy(path, t)
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

				//              Begin generate the test matrix A.
				//
				//              Set up parameters with DLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(matpath, imat, n, n)

				//              Generate a matrix with DLATMS.
				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)

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
						if uplo == Upper {
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
						if uplo == Upper {
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = min(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i1 = max(j, izero)
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
				for inb = 1; inb <= nnb; inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Dlacpy(uplo, n, n, a.Matrix(lda, opts), afac.Matrix(lda, opts))

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					lwork = max(2, nb) * lda
					*srnamt = "DsytrfRk"
					if info, err = golapack.DsytrfRk(uplo, n, afac.Matrix(lda, opts), e, &iwork, ainv, lwork); err != nil {
						panic(err)
					}

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

					//                 Check error code from DSYTRF_RK and handle error.
					if info != k {
						nerrs = alaerh(path, "DsytrfRk", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					result.Set(0, dsyt013(uplo, n, a.Matrix(lda, opts), afac.Matrix(lda, opts), e, iwork, ainv.Matrix(lda, opts), rwork))
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual,
					//                 if the factorization was competed without INFO > 0
					//                 (i.e. there is no zero rows and columns).
					//                 Do it only for the first block size.
					if inb == 1 && !trfcon {
						golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
						*srnamt = "Dsytri3"

						//                    Another reason that we need to compute the invesrse
						//                    is that DPOT03 produces RCONDC which is used later
						//                    in TEST6 and TEST7.
						lwork = (n + nb + 1) * (nb + 3)
						if info, err = golapack.Dsytri3(uplo, n, ainv.Matrix(lda, opts), e, &iwork, work, lwork); err != nil || info != 0 {
							nerrs = alaerh(path, "Dsytri3", info, -1, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						_result = result.GetPtr(1)
						rcondc, *_result = dpot03(uplo, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), work.Matrix(lda, opts), rwork)
						nt = 2
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" uplo=%s, n=%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//+    TEST 3
					//                 Compute largest element in U or L
					result.Set(2, zero)
					dtemp = zero

					_const = one / (one - alpha)

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
							dtemp = golapack.Dlange('M', k-1, 1, afac.MatrixOff(0+(k-1)*lda, lda, opts), rwork)
						} else {
							//
							//                       Get max absolute value from elements
							//                       in columns k and k-1 in U
							//
							dtemp = golapack.Dlange('M', k-2, 2, afac.MatrixOff(0+(k-2)*lda, lda, opts), rwork)
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
							dtemp = golapack.Dlange('M', n-k, 1, afac.MatrixOff(k+(k-1)*lda, lda, opts), rwork)
						} else {
							//                       Get max absolute value from elements
							//                       in columns k and k+1 in L
							dtemp = golapack.Dlange('M', n-k-1, 2, afac.MatrixOff(k+2-1+(k-1)*lda, lda, opts), rwork)
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

					_const = (one + alpha) / (one - alpha)
					golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))

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
							block.Set(0, 0, afac.Get(k-1-1+(k-2)*lda))
							block.Set(0, 1, e.Get(k-1))
							block.Set(1, 0, block.Get(0, 1))
							block.Set(1, 1, afac.Get(k-1+(k-1)*lda))

							if info, err = golapack.Dgesvd('N', 'N', 2, 2, block, rwork, ddummy, ddummy, work, 10); err != nil {
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
							block.Set(0, 0, afac.Get(k-1+(k-1)*lda))
							block.Set(1, 0, e.Get(k-1))
							block.Set(0, 1, block.Get(1, 0))
							block.Set(1, 1, afac.Get(k+(k)*lda))

							if info, err = golapack.Dgesvd('N', 'N', 2, 2, block, rwork, ddummy, ddummy, work, 10); err != nil {
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
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" uplo=%s, n=%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
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

						//+    TEST 5 ( Using TRS_3)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "Dlarhs"
						if err = Dlarhs(matpath, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

						*srnamt = "Dsytrs3"
						if info, err = golapack.Dsytrs3(uplo, n, nrhs, afac.Matrix(lda, opts), e, &iwork, x.Matrix(lda, opts)); err != nil || info != 0 {
							nerrs = alaerh(path, "Dsytrs3", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))

						//                    Compute the residual for the solution
						result.Set(4, dpot02(uplo, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

						//+    TEST 6
						//                    Check solution from generated exact solution.
						result.Set(5, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 5; k <= 6; k++ {
							if result.Get(k-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								t.Fail()
								fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun += 2

						//                 End do for each value of nrhs in NSVAL.
						//
					}

					//+    TEST 7
					//                 Get an estimate of RCOND = 1/CNDNUM.
				label230:
					;
					anorm = golapack.Dlansy('1', uplo, n, a.Matrix(lda, opts), rwork)
					*srnamt = "Dsycon3"
					if rcond, err = golapack.Dsycon3(uplo, n, afac.Matrix(lda, opts), e, &iwork, anorm, work, toSlice(&iwork, n)); err != nil {
						nerrs = alaerh(path, "Dsycon3", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					//                 Compute the test ratio to compare to values of RCOND
					result.Set(6, dget06(rcond, rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(6) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
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

	//     Verify number of tests match original.
	tgtRuns := 1618
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
