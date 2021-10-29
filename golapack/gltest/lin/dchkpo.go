package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkpo tests DPOTRF, -TRI, -TRS, -RFS, and -CON
func dchkpo(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var anorm, cndnum, rcond, rcondc, zero float64
	var _result *float64
	var i, imat, in, inb, info, ioff, irhs, izero, k, kl, ku, lda, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, ntypes int
	var err error

	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	ntypes = 9

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dpo"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrpo(path, t)
	}
	(*infot) = 0
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
				goto label110
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label110
			}

			//           Do first for uplo='U', then for uplo='L'
			for _, uplo = range mat.IterMatUplo(false) {

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)

				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
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
					if uplo == Upper {
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
				for inb = 1; inb <= nnb; inb++ {
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//                 Compute the L*L' or U'*U factorization of the matrix.
					golapack.Dlacpy(uplo, n, n, a.Matrix(lda, opts), afac.Matrix(lda, opts))
					*srnamt = "Dpotrf"
					if info, err = golapack.Dpotrf(uplo, n, afac.Matrix(lda, opts)); err != nil {
						panic(err)
					}

					//                 Check error code from DPOTRF.
					if info != izero {
						nerrs = alaerh(path, "Dpotrf", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
						goto label90
					}

					//                 Skip the tests if INFO is not 0.
					if info != 0 {
						goto label90
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
					result.Set(0, dpot01(uplo, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), rwork))

					//+    TEST 2
					//                 Form the inverse and compute the residual.
					golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
					*srnamt = "Dpotri"
					if info, err = golapack.Dpotri(uplo, n, ainv.Matrix(lda, opts)); err != nil {
						panic(err)
					}

					//                 Check error code from DPOTRI.
					if info != 0 {
						nerrs = alaerh(path, "Dpotri", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					_result = result.GetPtr(1)
					rcondc, *_result = dpot03(uplo, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), work.Matrix(lda, opts), rwork)

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= 2; k++ {
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

					//                 Skip the rest of the tests unless this is the first
					//                 blocksize.
					if inb != 1 {
						goto label90
					}

					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]

						//+    TEST 3
						//                 Solve and compute residual for A * X = B .
						*srnamt = "Dlarhs"
						if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

						*srnamt = "Dpotrs"
						if err = golapack.Dpotrs(uplo, n, nrhs, afac.Matrix(lda, opts), x.Matrix(lda, opts)); err != nil {
							nerrs = alaerh(path, "Dpotrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
						result.Set(2, dpot02(uplo, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

						//+    TEST 4
						//                 Check solution from generated exact solution.
						result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

						//+    TESTS 5, 6, and 7
						//                 Use iterative refinement to improve the solution.
						*srnamt = "Dporfs"
						if err = golapack.Dporfs(uplo, n, nrhs, a.Matrix(lda, opts), afac.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
							nerrs = alaerh(path, "Dporfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}
						//
						result.Set(4, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
						dpot05(uplo, n, nrhs, a.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(5))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 3; k <= 7; k++ {
							if result.Get(k-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								t.Fail()
								fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun += 5
					}

					//+    TEST 8
					//                 Get an estimate of RCOND = 1/CNDNUM.
					anorm = golapack.Dlansy('1', uplo, n, a.Matrix(lda, opts), rwork)
					*srnamt = "Dpocon"
					if rcond, err = golapack.Dpocon(uplo, n, afac.Matrix(lda, opts), anorm, work, &iwork); err != nil {
						nerrs = alaerh(path, "Dpocon", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(7, dget06(rcond, rcondc))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(7) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" uplo=%s, n=%5d,           _type %2d, test(%2d) =%12.5f\n", uplo, n, imat, 8, result.Get(7))
						nfail++
					}
					nrun++
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
	alasum(path, nfail, nrun, nerrs)
}
