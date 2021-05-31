package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkpo tests ZPOTRF, -TRI, -TRS, -RFS, and -CON
func Zchkpo(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo, xtype byte
	var czero complex128
	var anorm, cndnum, rcond, rcondc float64
	var i, imat, in, inb, info, ioff, irhs, iuplo, izero, k, kl, ku, lda, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, ntypes int

	uplos := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	czero = (0.0 + 0.0*1i)
	ntypes = 9
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("ZPO")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrpo(path, t)
	}
	(*infot) = 0

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

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with ZLATMS.
				Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS.
				if info != 0 {
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
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
					izero = 0
				}

				//              Set the imaginary part of the diagonals.
				Zlaipd(&n, a, toPtr(lda+1), func() *int { y := 0; return &y }())

				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= (*nnb); inb++ {
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Compute the L*L' or U'*U factorization of the matrix.
					golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)
					*srnamt = "ZPOTRF"
					golapack.Zpotrf(uplo, &n, afac.CMatrix(lda, opts), &lda, &info)

					//                 Check error code from ZPOTRF.
					if info != izero {
						Alaerh(path, []byte("ZPOTRF"), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
						goto label90
					}

					//                 Skip the tests if INFO is not 0.
					if info != 0 {
						goto label90
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Zlacpy(uplo, &n, &n, afac.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
					Zpot01(uplo, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))

					//+    TEST 2
					//                 Form the inverse and compute the residual.
					golapack.Zlacpy(uplo, &n, &n, afac.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
					*srnamt = "ZPOTRI"
					golapack.Zpotri(uplo, &n, ainv.CMatrix(lda, opts), &lda, &info)

					//                 Check error code from ZPOTRI.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZPOTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Zpot03(uplo, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= 2; k++ {
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

					//                 Skip the rest of the tests unless this is the first
					//                 blocksize.
					if inb != 1 {
						goto label90
					}

					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]

						//+    TEST 3
						//                 Solve and compute residual for A * X = B .
						*srnamt = "ZLARHS"
						Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZPOTRS"
						golapack.Zpotrs(uplo, &n, &nrhs, afac.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &info)

						//                 Check error code from ZPOTRS.
						if info != 0 {
							Alaerh(path, []byte("ZPOTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
						Zpot02(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(2))

						//+    TEST 4
						//                 Check solution from generated exact solution.
						Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

						//+    TESTS 5, 6, and 7
						//                 Use iterative refinement to improve the solution.
						*srnamt = "ZPORFS"
						golapack.Zporfs(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

						//                 Check error code from ZPORFS.
						if info != 0 {
							Alaerh(path, []byte("ZPORFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(4))
						Zpot05(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(5))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 7; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + 5
					}

					//+    TEST 8
					//                 Get an estimate of RCOND = 1/CNDNUM.
					anorm = golapack.Zlanhe('1', uplo, &n, a.CMatrix(lda, opts), &lda, rwork)
					*srnamt = "ZPOCON"
					golapack.Zpocon(uplo, &n, afac.CMatrix(lda, opts), &lda, &anorm, &rcond, work, rwork, &info)

					//                 Check error code from ZPOCON.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZPOCON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					result.Set(7, Dget06(&rcond, &rcondc))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(7) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
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

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
