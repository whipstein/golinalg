package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkhe tests Zhetrf, -TRI2, -TRS, -TRS2, -RFS, and -CON.
func zchkhe(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var czero complex128
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	ntypes = 10
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Zhe"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrhe(path, t)
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

				//              Set up parameters with ZLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

				//              Generate a matrix with Zlatms.
				*srnamt = "Zlatms"
				if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.CMatrix(lda, opts), work); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)

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
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = min(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, czero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
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

				//              End generate test matrix A.
				//
				//
				//              Set the imaginary part of the diagonals.
				zlaipd(n, a, lda+1, 0)

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
					*srnamt = "Zhetrf"
					info, err = golapack.Zhetrf(uplo, n, afac.CMatrix(lda, opts), &iwork, ainv, lwork)

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

					//                 Check error code from Zhetrf and handle error.
					if err != nil || info != k {
						t.Fail()
						nerrs = alaerh(path, "Zhetrf", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Set the condition estimate flag if the INFO is not 0.
					if info != 0 {
						trfcon = true
					} else {
						trfcon = false
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					*result.GetPtr(0) = zhet01(uplo, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, ainv.CMatrix(lda, opts), rwork)
					nt = 1

					//+    TEST 2
					//                 Form the inverse and compute the residual.
					if inb == 1 && !trfcon {
						golapack.Zlacpy(uplo, n, n, afac.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
						*srnamt = "Zhetri2"
						lwork = (n + nb + 1) * (nb + 3)
						if info, err = golapack.Zhetri2(uplo, n, ainv.CMatrix(lda, opts), &iwork, work, lwork); err != nil || info != 0 {
							t.Fail()
							nerrs = alaerh(path, "Zhetri", info, -1, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						//                    Compute the residual for a symmetric matrix times
						//                    its inverse.
						rcondc, *result.GetPtr(1) = zpot03(uplo, n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)
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

						//+    TEST 3 (Using TRS)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "zlarhs"
						if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "Zhetrs"
						if err = golapack.Zhetrs(uplo, n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts)); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zhetrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))

						//                    Compute the residual for the solution
						*result.GetPtr(2) = zpot02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

						//+    TEST 4 (Using TRS2)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of nrhs random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "zlarhs"
						if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "Zhetrs2"
						if err = golapack.Zhetrs2(uplo, n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts), work); err != nil {
							panic(err)
						}

						//                    Check error code from Zhetrs2 and handle error.
						if info != 0 {
							t.Fail()
							nerrs = alaerh(path, "Zhetrs2", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))

						//                    Compute the residual for the solution
						*result.GetPtr(3) = zpot02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

						//+    TEST 5
						//                 Check solution from generated exact solution.
						*result.GetPtr(4) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

						//+    TESTS 6, 7, and 8
						//                 Use iterative refinement to improve the solution.
						*srnamt = "Zherfs"
						if err = golapack.Zherfs(uplo, n, nrhs, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zherfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						*result.GetPtr(5) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
						zpot05(uplo, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(6))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 8; k++ {
							if result.Get(k-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
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
					anorm = golapack.Zlanhe('1', uplo, n, a.CMatrix(lda, opts), rwork)
					*srnamt = "Zhecon"
					if rcond, err = golapack.Zhecon(uplo, n, afac.CMatrix(lda, opts), &iwork, anorm, work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zhecon", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(8, dget06(rcond, rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(8) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
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

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
