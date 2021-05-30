package lin

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Dchkpp tests DPPTRF, -TRI, -TRS, -RFS, and -CON
func Dchkpp(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, packit, _type, uplo, xtype byte
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, imat, in, info, ioff, irhs, iuplo, izero, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, npp, nrhs, nrun, ntypes int

	packs := make([]byte, 2)
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
	uplos[0], uplos[1], packs[0], packs[1] = 'U', 'L', 'C', 'R'

	//     Initialize constants and the random number seed.
	path := []byte("DPP")
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

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label100
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label100
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]
				packit = packs[iuplo-1]

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label90
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

					//                 Set row and column IZERO of A to 0.
					if iuplo == 1 {
						ioff = (izero - 1) * izero / 2
						for i = 1; i <= izero-1; i++ {
							a.Set(ioff+i-1, zero)
						}
						ioff = ioff + izero
						for i = izero; i <= n; i++ {
							a.Set(ioff-1, zero)
							ioff = ioff + i
						}
					} else {
						ioff = izero
						for i = 1; i <= izero-1; i++ {
							a.Set(ioff-1, zero)
							ioff = ioff + n - i
						}
						ioff = ioff - izero
						for i = izero; i <= n; i++ {
							a.Set(ioff+i-1, zero)
						}
					}
				} else {
					izero = 0
				}

				//              Compute the L*L' or U'*U factorization of the matrix.
				npp = n * (n + 1) / 2
				goblas.Dcopy(&npp, a, toPtr(1), afac, toPtr(1))
				*srnamt = "DPPTRF"
				golapack.Dpptrf(uplo, &n, afac, &info)

				//              Check error code from DPPTRF.
				if info != izero {
					Alaerh(path, []byte("DPPTRF"), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label90
				}

				//              Skip the tests if INFO is not 0.
				if info != 0 {
					goto label90
				}

				//+    TEST 1
				//              Reconstruct matrix from factors and compute residual.
				goblas.Dcopy(&npp, afac, toPtr(1), ainv, toPtr(1))
				Dppt01(uplo, &n, a, ainv, rwork, result.GetPtr(0))

				//+    TEST 2
				//              Form the inverse and compute the residual.
				goblas.Dcopy(&npp, afac, toPtr(1), ainv, toPtr(1))
				*srnamt = "DPPTRI"
				golapack.Dpptri(uplo, &n, ainv, &info)

				//              Check error code from DPPTRI.
				if info != 0 {
					Alaerh(path, []byte("DPPTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				Dppt03(uplo, &n, a, ainv, work.Matrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = 1; k <= 2; k++ {
					if result.Get(k-1) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
				}
				nrun = nrun + 2

				for irhs = 1; irhs <= (*nns); irhs++ {
					nrhs = (*nsval)[irhs-1]
					//+    TEST 3
					//              Solve and compute residual for  A * X = B.
					*srnamt = "DLARHS"
					Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
					golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

					*srnamt = "DPPTRS"
					golapack.Dpptrs(uplo, &n, &nrhs, afac, x.Matrix(lda, opts), &lda, &info)

					//              Check error code from DPPTRS.
					if info != 0 {
						Alaerh(path, []byte("DPPTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
					Dppt02(uplo, &n, &nrhs, a, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(2))

					//+    TEST 4
					//              Check solution from generated exact solution.
					Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

					//+    TESTS 5, 6, and 7
					//              Use iterative refinement to improve the solution.
					*srnamt = "DPPRFS"
					golapack.Dpprfs(uplo, &n, &nrhs, a, afac, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, iwork, &info)

					//              Check error code from DPPRFS.
					if info != 0 {
						Alaerh(path, []byte("DPPRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(4))
					Dppt05(uplo, &n, &nrhs, a, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(5))

					//                 Print information about the tests that did not pass
					//                 the threshold.
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
				//              Get an estimate of RCOND = 1/CNDNUM.
				anorm = golapack.Dlansp('1', uplo, &n, a, rwork)
				*srnamt = "DPPCON"
				golapack.Dppcon(uplo, &n, afac, &anorm, &rcond, work, iwork, &info)

				//              Check error code from DPPCON.
				if info != 0 {
					Alaerh(path, []byte("DPPCON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				result.Set(7, Dget06(&rcond, &rcondc))

				//              Print the test ratio if greater than or equal to THRESH.
				if result.Get(7) >= (*thresh) {
					if nfail == 0 && nerrs == 0 {
						Alahd(path)
					}
					t.Fail()
					fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, 8, result.Get(7))
					nfail = nfail + 1
				}
				nrun = nrun + 1
			label90:
			}
		label100:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1332
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
