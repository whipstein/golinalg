package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dchkpo tests DPOTRF, -TRI, -TRS, -RFS, and -CON
func Dchkpo(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo, xtype byte
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, imat, in, inb, info, ioff, irhs, iuplo, izero, k, kl, ku, lda, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, ntypes int

	uplos := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	ntypes = 9
	// ntests = 8

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("DPO")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrpo(path, t)
	}
	(*infot) = 0
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
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label110
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label110
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label100
				}

				//              For types 3-5, zero one row and column of the matrix to
				//              test that INFO is returned correctly.
				if zerot {
					if imat == 3 {
						izero = 1
					} else if imat == 4 {
						izero = n
					} else {
						izero = n/2 + 1
					}
					ioff = (izero - 1) * lda

					//                 Set row and column IZERO of A to 0.
					if iuplo == 1 {
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
					izero = 0
				}

				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= (*nnb); inb++ {
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Compute the L*L' or U'*U factorization of the matrix.
					golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
					*srnamt = "DPOTRF"
					golapack.Dpotrf(uplo, &n, afac.Matrix(lda, opts), &lda, &info)

					//                 Check error code from DPOTRF.
					if info != izero {
						Alaerh(path, []byte("DPOTRF"), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
						goto label90
					}

					//                 Skip the tests if INFO is not 0.
					if info != 0 {
						goto label90
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
					Dpot01(uplo, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))

					//+    TEST 2
					//                 Form the inverse and compute the residual.
					golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
					*srnamt = "DPOTRI"
					golapack.Dpotri(uplo, &n, ainv.Matrix(lda, opts), &lda, &info)

					//                 Check error code from DPOTRI.
					if info != 0 {
						Alaerh(path, []byte("DPOTRI"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dpot03(uplo, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= 2; k++ {
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

					//                 Skip the rest of the tests unless this is the first
					//                 blocksize.
					if inb != 1 {
						goto label90
					}

					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]

						//+    TEST 3
						//                 Solve and compute residual for A * X = B .
						*srnamt = "DLARHS"
						Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						*srnamt = "DPOTRS"
						golapack.Dpotrs(uplo, &n, &nrhs, afac.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &info)

						//                 Check error code from DPOTRS.
						if info != 0 {
							Alaerh(path, []byte("DPOTRS"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
						Dpot02(uplo, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(2))

						//+    TEST 4
						//                 Check solution from generated exact solution.
						Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

						//+    TESTS 5, 6, and 7
						//                 Use iterative refinement to improve the solution.
						*srnamt = "DPORFS"
						golapack.Dporfs(uplo, &n, &nrhs, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), work, iwork, &info)

						//                 Check error code from DPORFS.
						if info != 0 {
							Alaerh(path, []byte("DPORFS"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}
						//
						Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(4))
						Dpot05(uplo, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), result.Off(5))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 7; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								t.Fail()
								fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + 5
					}

					//+    TEST 8
					//                 Get an estimate of RCOND = 1/CNDNUM.
					anorm = golapack.Dlansy('1', uplo, &n, a.Matrix(lda, opts), &lda, rwork)
					*srnamt = "DPOCON"
					golapack.Dpocon(uplo, &n, afac.Matrix(lda, opts), &lda, &anorm, &rcond, work, iwork, &info)

					//                 Check error code from DPOCON.
					if info != 0 {
						Alaerh(path, []byte("DPOCON"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					result.Set(7, Dget06(&rcond, &rcondc))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(7) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" UPLO = '%c', N =%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 8, result.Get(7))
						nfail = nfail + 1
					}
					nrun = nrun + 1
				label90:
				}
			label100:
			}
		label110:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1628
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
