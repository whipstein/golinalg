package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchksy tests DSYTRF, -TRI2, -TRS, -TRS2, -RFS, and -CON.
func dchksy(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var anorm, cndnum, rcond, rcondc, zero float64
	var _result *float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	ntypes = 10

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dsy"
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
				goto label170
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label170
			}

			//           Do first for uplo='U', then for uplo='L'
			for _, uplo = range mat.IterMatUplo(false) {
				//              Begin generate the test matrix A.
				//
				//
				//              Set up parameters with DLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)

				//              Generate a matrix with DLATMS.
				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)

					//                    Skip all tests for this generated matrix
					goto label160
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
					*srnamt = "Dsytrf"
					if info, err = golapack.Dsytrf(uplo, n, afac.Matrix(lda, opts), &iwork, ainv, lwork); err != nil {
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

					//                 Check error code from DSYTRF and handle error.
					if info != k {
						nerrs = alaerh(path, "Dsytrf", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					result.Set(0, dsyt01(uplo, n, a.Matrix(lda, opts), afac.Matrix(lda, opts), iwork, ainv.Matrix(lda, opts), rwork))
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual,
					//                 if the factorization was competed without INFO > 0
					//                 (i.e. there is no zero rows and columns).
					//                 Do it only for the first block size.
					if inb == 1 && !trfcon {
						golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
						*srnamt = "Dsytri2"
						lwork = (n + nb + 1) * (nb + 3)
						if info, err = golapack.Dsytri2(uplo, n, ainv.Matrix(lda, opts), &iwork, work.Matrix(lda, opts), lwork); err != nil || info != 0 {
							nerrs = alaerh(path, "Dsytri2", info, -1, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
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
							fmt.Printf(" uplo=%s, n=%5d, nb=%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Skip the other tests if this is not the first block
					//                 size.
					if inb > 1 {
						goto label150
					}

					//                 Do only the condition estimate if INFO is not 0.
					if trfcon {
						rcondc = zero
						goto label140
					}

					//                 Do for each value of nrhs in NSVAL.
					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]

						//+    TEST 3 ( Using TRS)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "Dlarhs"
						if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

						*srnamt = "Dsytrs"
						if err = golapack.Dsytrs(uplo, n, nrhs, afac.Matrix(lda, opts), &iwork, x.Matrix(lda, opts)); err != nil {
							nerrs = alaerh(path, "Dsytrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))

						//                    Compute the residual for the solution
						result.Set(2, dpot02(uplo, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

						//+    TEST 4 (Using TRS2)
						//
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "Dlarhs"
						if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

						*srnamt = "Dsytrs2"
						if err = golapack.Dsytrs2(uplo, n, nrhs, afac.Matrix(lda, opts), &iwork, x.Matrix(lda, opts), work); err != nil {
							nerrs = alaerh(path, "Dsytrs2", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))

						//                    Compute the residual for the solution
						result.Set(3, dpot02(uplo, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

						//+    TEST 5
						//                 Check solution from generated exact solution.
						result.Set(4, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

						//+    TESTS 6, 7, and 8
						//                 Use iterative refinement to improve the solution.
						*srnamt = "Dsyrfs"
						if info, err = golapack.Dsyrfs(uplo, n, nrhs, a.Matrix(lda, opts), afac.Matrix(lda, opts), &iwork, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, toSlice(&iwork, n)); err != nil || info != 0 {
							nerrs = alaerh(path, "Dsyrfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						result.Set(5, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
						dpot05(uplo, n, nrhs, a.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(6))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 8; k++ {
							if result.Get(k-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								t.Fail()
								fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun += 6

						//                 End do for each value of nrhs in NSVAL.

					}

					//+    TEST 9
					//                 Get an estimate of RCOND = 1/CNDNUM.
				label140:
					;
					anorm = golapack.Dlansy('1', uplo, n, a.Matrix(lda, opts), rwork)
					*srnamt = "Dsycon"
					if rcond, err = golapack.Dsycon(uplo, n, afac.Matrix(lda, opts), &iwork, anorm, work, toSlice(&iwork, n)); err != nil {
						nerrs = alaerh(path, "Dsycon", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					//                 Compute the test ratio to compare values of RCOND
					result.Set(8, dget06(rcond, rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(8) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" uplo=%s, n=%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 9, result.Get(8))
						nfail++
					}
					nrun++
				label150:
				}

			label160:
			}
		label170:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1846
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
