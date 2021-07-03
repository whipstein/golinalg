package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dchksp tests DSPTRF, -TRI, -TRS, -RFS, and -CON
func Dchksp(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, packit, _type, uplo, xtype byte
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, i1, i2, imat, in, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, npp, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	ntypes = 10

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("DSP")
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
				goto label160
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label160
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]
				if uplo == 'U' {
					packit = 'C'
				} else {
					packit = 'R'
				}

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label150
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
						ioff = 0
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i2 = minint(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + j
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i1 = maxint(j, izero)
								for i = i1; i <= n; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + n - j
							}
						}
					}
				} else {
					izero = 0
				}

				//              Compute the L*D*L' or U*D*U' factorization of the matrix.
				npp = n * (n + 1) / 2
				goblas.Dcopy(npp, a, 1, afac, 1)
				*srnamt = "DSPTRF"
				golapack.Dsptrf(uplo, &n, afac, iwork, &info)

				//              Adjust the expected value of INFO to account for
				//              pivoting.
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

				//              Check error code from DSPTRF.
				if info != k {
					Alaerh(path, []byte("DSPTRF"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}
				if info != 0 {
					trfcon = true
				} else {
					trfcon = false
				}

				//+    TEST 1
				//              Reconstruct matrix from factors and compute residual.
				Dspt01(uplo, &n, a, afac, iwork, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))
				nt = 1

				//+    TEST 2
				//              Form the inverse and compute the residual.
				if !trfcon {
					goblas.Dcopy(npp, afac, 1, ainv, 1)
					*srnamt = "DSPTRI"
					golapack.Dsptri(uplo, &n, ainv, iwork, work, &info)

					//              Check error code from DSPTRI.
					if info != 0 {
						Alaerh(path, []byte("DSPTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dppt03(uplo, &n, a, ainv, work.Matrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))
					nt = 2
				}

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = 1; k <= nt; k++ {
					if result.Get(k-1) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
				}
				nrun = nrun + nt

				//              Do only the condition estimate if INFO is not 0.
				if trfcon {
					rcondc = zero
					goto label140
				}

				for irhs = 1; irhs <= (*nns); irhs++ {
					nrhs = (*nsval)[irhs-1]

					//+    TEST 3
					//              Solve and compute residual for  A * X = B.
					*srnamt = "DLARHS"
					Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
					golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

					*srnamt = "DSPTRS"
					golapack.Dsptrs(uplo, &n, &nrhs, afac, iwork, x.Matrix(lda, opts), &lda, &info)

					//              Check error code from DSPTRS.
					if info != 0 {
						Alaerh(path, []byte("DSPTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}
					//
					golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
					Dppt02(uplo, &n, &nrhs, a, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(2))

					//+    TEST 4
					//              Check solution from generated exact solution.
					Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

					//+    TESTS 5, 6, and 7
					//              Use iterative refinement to improve the solution.
					*srnamt = "DSPRFS"
					golapack.Dsprfs(uplo, &n, &nrhs, a, afac, iwork, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, toSlice(iwork, n+1-1), &info)

					//              Check error code from DSPRFS.
					if info != 0 {
						Alaerh(path, []byte("DSPRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
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
			label140:
				;
				anorm = golapack.Dlansp('1', uplo, &n, a, rwork)
				*srnamt = "DSPCON"
				golapack.Dspcon(uplo, &n, afac, iwork, &anorm, &rcond, work, toSlice(iwork, n+1-1), &info)

				//              Check error code from DSPCON.
				if info != 0 {
					Alaerh(path, []byte("DSPCON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				result.Set(7, Dget06(&rcond, &rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(7) >= (*thresh) {
					if nfail == 0 && nerrs == 0 {
						Alahd(path)
					}
					t.Fail()
					fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, 8, result.Get(7))
					nfail = nfail + 1
				}
				nrun = nrun + 1
			label150:
			}
		label160:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1404
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
