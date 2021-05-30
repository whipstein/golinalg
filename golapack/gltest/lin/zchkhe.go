package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Zchkhe tests ZHETRF, -TRI2, -TRS, -TRS2, -RFS, and -CON.
func Zchkhe(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, uplo, xtype byte
	var czero complex128
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	ntypes = 10
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("ZHE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrhe(path, t)
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
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label170
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label170
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]

				//              Set up parameters with ZLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				//              Generate a matrix with ZLATMS.
				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS and handle error.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)

					//                 Skip all tests for this generated matrix
					goto label160
				}

				//              For types 3-6, zero one or more rows and columns of
				//              the matrix to test that INFO is returned correctly.
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
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = minint(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, czero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i1 = maxint(j, izero)
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

				//              End generate test matrix A.
				//
				//
				//              Set the imaginary part of the diagonals.
				Zlaipd(&n, a, toPtr(lda+1), func() *int { y := 0; return &y }())

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
					lwork = maxint(2, nb) * lda
					*srnamt = "ZHETRF"
					golapack.Zhetrf(uplo, &n, afac.CMatrix(lda, opts), &lda, iwork, ainv, &lwork, &info)

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

					//                 Check error code from ZHETRF and handle error.
					if info != k {
						t.Fail()
						Alaerh(path, []byte("ZHETRF"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					Zhet01(uplo, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, ainv.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual.
					if inb == 1 && !trfcon {
						golapack.Zlacpy(uplo, &n, &n, afac.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
						*srnamt = "ZHETRI2"
						lwork = (n + nb + 1) * (nb + 3)
						golapack.Zhetri2(uplo, &n, ainv.CMatrix(lda, opts), &lda, iwork, work, &lwork, &info)

						//                    Check error code from ZHETRI and handle error.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZHETRI"), &info, toPtr(-1), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						Zpot03(uplo, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))
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

					//                 Do for each value of NRHS in NSVAL.
					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]

						//+    TEST 3 (Using TRS)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of NRHS random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "ZLARHS"
						Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZHETRS"
						golapack.Zhetrs(uplo, &n, &nrhs, afac.CMatrix(lda, opts), &lda, iwork, x.CMatrix(lda, opts), &lda, &info)

						//                    Check error code from ZHETRS and handle error.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZHETRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)

						//                    Compute the residual for the solution
						Zpot02(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(2))

						//+    TEST 4 (Using TRS2)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of NRHS random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "ZLARHS"
						Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZHETRS2"
						golapack.Zhetrs2(uplo, &n, &nrhs, afac.CMatrix(lda, opts), &lda, iwork, x.CMatrix(lda, opts), &lda, work, &info)

						//                    Check error code from ZHETRS2 and handle error.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZHETRS2"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)

						//                    Compute the residual for the solution
						Zpot02(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(3))

						//+    TEST 5
						//                 Check solution from generated exact solution.
						Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(4))

						//+    TESTS 6, 7, and 8
						//                 Use iterative refinement to improve the solution.
						*srnamt = "ZHERFS"
						golapack.Zherfs(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

						//                    Check error code from ZHERFS.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZHERFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(5))
						Zpot05(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(6))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 8; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + 6

						//                 End do for each value of NRHS in NSVAL.
					}

					//+    TEST 9
					//                 Get an estimate of RCOND = 1/CNDNUM.
				label140:
					;
					anorm = golapack.Zlanhe('1', uplo, &n, a.CMatrix(lda, opts), &lda, rwork)
					*srnamt = "ZHECON"
					golapack.Zhecon(uplo, &n, afac.CMatrix(lda, opts), &lda, iwork, &anorm, &rcond, work, &info)

					//                 Check error code from ZHECON and handle error.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZHECON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					result.Set(8, Dget06(&rcond, &rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(8) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" UPLO = '%c', N =%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 9, result.Get(8))
						nfail = nfail + 1
					}
					nrun = nrun + 1
				label150:
				}
			label160:
			}
		label170:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
