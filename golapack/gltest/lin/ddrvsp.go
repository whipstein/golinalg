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

// Ddrvsp tests the driver routines DSPSV and -SVX.
func Ddrvsp(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, fact, packit, _type, uplo, xtype byte
	var ainvnm, anorm, cndnum, one, rcond, rcondc, zero float64
	var i, i1, i2, ifact, imat, in, info, ioff, iuplo, izero, j, k, k1, kl, ku, lda, mode, n, nerrs, nfact, nfail, nimat, npp, nrun, nt, ntypes int

	facts := make([]byte, 2)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 10
	nfact = 2

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1] = 'F', 'N'

	//     Initialize constants and the random number seed.
	path := []byte("DSP")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	// lwork = maxint(2*(*nmax), (*nmax)*(*nrhs))

	//     Test the error exits
	if *tsterr {
		Derrvx(path, t)
	}
	(*infot) = 0

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
		npp = n * (n + 1) / 2
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

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
				if iuplo == 1 {
					uplo = 'U'
					packit = 'C'
				} else {
					uplo = 'L'
					packit = 'R'
				}

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.Matrix(lda, opts), &lda, work, &info)
				//
				//              Check error code from DLATMS.
				//
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label160
				}

				//              For types 3-6, zero one or more rows and columns of the
				//              matrix to test that INFO is returned correctly.
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

				for ifact = 1; ifact <= nfact; ifact++ {
					//                 Do first for FACT = 'F', then for other values.
					fact = facts[ifact-1]

					//                 Compute the condition number for comparison with
					//                 the value returned by DSPSVX.
					if zerot {
						if ifact == 1 {
							goto label150
						}
						rcondc = zero

					} else if ifact == 1 {
						//                    Compute the 1-norm of A.
						anorm = golapack.Dlansp('1', uplo, &n, a, rwork)

						//                    Factor the matrix A.
						goblas.Dcopy(npp, a, 1, afac, 1)
						golapack.Dsptrf(uplo, &n, afac, iwork, &info)

						//                    Compute inv(A) and take its norm.
						goblas.Dcopy(npp, afac, 1, ainv, 1)
						golapack.Dsptri(uplo, &n, ainv, iwork, work, &info)
						ainvnm = golapack.Dlansp('1', uplo, &n, ainv, rwork)

						//                    Compute the 1-norm condition number of A.
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}
					}

					//                 Form an exact solution and set the right hand side.
					*srnamt = "DLARHS"
					Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
					xtype = 'C'

					//                 --- Test DSPSV  ---
					if ifact == 2 {
						goblas.Dcopy(npp, a, 1, afac, 1)
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						//                    Factor the matrix and solve the system using DSPSV.
						*srnamt = "DSPSV "
						golapack.Dspsv(uplo, &n, nrhs, afac, iwork, x.Matrix(lda, opts), &lda, &info)

						//                    Adjust the expected value of INFO to account for
						//                    pivoting.
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

						//                    Check error code from DSPSV .
						if info != k {
							Alaerh(path, []byte("DSPSV "), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label120
						} else if info != 0 {
							goto label120
						}

						//                    Reconstruct matrix from factors and compute
						//                    residual.
						Dspt01(uplo, &n, a, afac, iwork, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))

						//                    Compute residual of the computed solution.
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
						Dppt02(uplo, &n, nrhs, a, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
						nt = 3

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 1; k <= nt; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								fmt.Printf(" %s, UPLO='%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", "DSPSV ", uplo, n, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt
					label120:
					}

					//                 --- Test DSPSVX ---
					if ifact == 2 && npp > 0 {
						golapack.Dlaset('F', &npp, toPtr(1), &zero, &zero, afac.Matrix(npp, opts), &npp)
					}
					golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(lda, opts), &lda)

					//                 Solve the system and compute the condition number and
					//                 error bounds using DSPSVX.
					*srnamt = "DSPSVX"
					golapack.Dspsvx(fact, uplo, &n, nrhs, a, afac, iwork, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, toSlice(iwork, n+1-1), &info)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					k = izero
					if k > 0 {
					label130:
						;
						if (*iwork)[k-1] < 0 {
							if (*iwork)[k-1] != -k {
								k = -(*iwork)[k-1]
								goto label130
							}
						} else if (*iwork)[k-1] != k {
							k = (*iwork)[k-1]
							goto label130
						}
					}

					//                 Check the error code from DSPSVX.
					if info != k {
						Alaerh(path, []byte("DSPSVX"), &info, &k, []byte{fact, uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
						goto label150
					}

					if info == 0 {
						if ifact >= 2 {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dspt01(uplo, &n, a, afac, iwork, ainv.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(0))
							k1 = 1
						} else {
							k1 = 2
						}

						//                    Compute residual of the computed solution.
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
						Dppt02(uplo, &n, nrhs, a, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

						//                    Check the error bounds from iterative refinement.
						Dppt05(uplo, &n, nrhs, a, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)+1-1), result.Off(3))
					} else {
						k1 = 6
					}

					//                 Compare RCOND from DSPSVX with the computed value
					//                 in RCONDC.
					result.Set(5, Dget06(&rcond, &rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= 6; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Aladhd(path)
							}
							t.Fail()
							fmt.Printf(" %s, FACT='%c', UPLO='%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", "DSPSVX", fact, uplo, n, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 7 - k1

				label150:
				}

			label160:
			}
		label170:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1072
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
