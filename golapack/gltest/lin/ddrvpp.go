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

// Ddrvpp tests the driver routines DPPSV and -SVX.
func Ddrvpp(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var equil, nofact, prefac, zerot bool
	var dist, equed, fact, packit, _type, uplo, xtype byte
	var ainvnm, amax, anorm, cndnum, one, rcond, rcondc, roldc, scond, zero float64
	var i, iequed, ifact, imat, in, info, ioff, iuplo, izero, k, k1, kl, ku, lda, mode, n, nerrs, nfact, nfail, nimat, npp, nrun, nt, ntypes int

	equeds := make([]byte, 2)
	facts := make([]byte, 3)
	packs := make([]byte, 2)
	uplos := make([]byte, 2)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 9
	// ntests = 6

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], facts[0], facts[1], facts[2], packs[0], packs[1], equeds[0], equeds[1] = 'U', 'L', 'F', 'N', 'E', 'C', 'R', 'N', 'Y'

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
				goto label130
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label130
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]
				packit = packs[iuplo-1]

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
				rcondc = one / cndnum

				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.Matrix(lda, opts), &lda, work, &info)
				//
				//              Check error code from DLATMS.
				//
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label120
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

				//              Save a copy of the matrix A in ASAV.
				goblas.Dcopy(&npp, a.Off(0), toPtr(1), asav.Off(0), toPtr(1))

				for iequed = 1; iequed <= 2; iequed++ {
					equed = equeds[iequed-1]
					if iequed == 1 {
						nfact = 3
					} else {
						nfact = 1
					}
					//
					for ifact = 1; ifact <= nfact; ifact++ {
						fact = facts[ifact-1]
						prefac = fact == 'F'
						nofact = fact == 'N'
						equil = fact == 'E'

						if zerot {
							if prefac {
								goto label100
							}
							rcondc = zero

						} else if fact != 'N' {
							//                       Compute the condition number for comparison with
							//                       the value returned by DPPSVX (FACT = 'N' reuses
							//                       the condition number from the previous iteration
							//                       with FACT = 'F').
							goblas.Dcopy(&npp, asav.Off(0), toPtr(1), afac.Off(0), toPtr(1))
							if equil || iequed > 1 {
								//                          Compute row and column scale factors to
								//                          equilibrate the matrix A.
								golapack.Dppequ(uplo, &n, afac, s, &scond, &amax, &info)
								if info == 0 && n > 0 {
									if iequed > 1 {
										scond = zero
									}

									//                             Equilibrate the matrix.
									golapack.Dlaqsp(uplo, &n, afac, s, &scond, &amax, &equed)
								}
							}

							//                       Save the condition number of the
							//                       non-equilibrated system for use in DGET04.
							if equil {
								roldc = rcondc
							}

							//                       Compute the 1-norm of A.
							anorm = golapack.Dlansp('1', uplo, &n, afac, rwork)

							//                       Factor the matrix A.
							golapack.Dpptrf(uplo, &n, afac, &info)

							//                       Form the inverse of A.
							goblas.Dcopy(&npp, afac.Off(0), toPtr(1), a.Off(0), toPtr(1))
							golapack.Dpptri(uplo, &n, a, &info)

							//                       Compute the 1-norm condition number of A.
							ainvnm = golapack.Dlansp('1', uplo, &n, a, rwork)
							if anorm <= zero || ainvnm <= zero {
								rcondc = one
							} else {
								rcondc = (one / anorm) / ainvnm
							}
						}

						//                    Restore the matrix A.
						goblas.Dcopy(&npp, asav, toPtr(1), a, toPtr(1))

						//                    Form an exact solution and set the right hand side.
						*srnamt = "DLARHS"
						Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, bsav.Matrix(lda, opts), &lda)

						if nofact {
							//                       --- Test DPPSV  ---
							//
							//                       Compute the L*L' or U'*U factorization of the
							//                       matrix and solve the system.
							goblas.Dcopy(&npp, a, toPtr(1), afac, toPtr(1))
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

							*srnamt = "DPPSV "
							golapack.Dppsv(uplo, &n, nrhs, afac, x.Matrix(lda, opts), &lda, &info)

							//                       Check error code from DPPSV .
							if info != izero {
								Alaerh(path, []byte("DPPSV "), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
								goto label70
							} else if info != 0 {
								goto label70
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dppt01(uplo, &n, a, afac, rwork, result.GetPtr(0))

							//                       Compute residual of the computed solution.
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dppt02(uplo, &n, nrhs, a, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

							//                       Check solution from generated exact solution.
							Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							nt = 3

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									t.Fail()
									fmt.Printf(" %s, UPLO='%c', N =%5d, _type %1d, test(%1d)=%12.5f\n", "DPPSV ", uplo, n, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + nt
						label70:
						}

						//                    --- Test DPPSVX ---
						if !prefac && npp > 0 {
							golapack.Dlaset('F', &npp, toPtr(1), &zero, &zero, afac.Matrix(npp, opts), &npp)
						}
						golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(lda, opts), &lda)
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT='F' and
							//                       EQUED='Y'.
							golapack.Dlaqsp(uplo, &n, a, s, &scond, &amax, &equed)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using DPPSVX.
						*srnamt = "DPPSVX"
						golapack.Dppsvx(fact, uplo, &n, nrhs, a, afac, &equed, s, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, iwork, &info)

						//                    Check the error code from DPPSVX.
						if info != izero {
							Alaerh(path, []byte("DPPSVX"), &info, &izero, []byte{fact, uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label90
						}

						if info == 0 {
							if !prefac {
								//                          Reconstruct matrix from factors and compute
								//                          residual.
								Dppt01(uplo, &n, a, afac, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(0))
								k1 = 1
							} else {
								k1 = 2
							}

							//                       Compute residual of the computed solution.
							golapack.Dlacpy('F', &n, nrhs, bsav.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dppt02(uplo, &n, nrhs, asav, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(1))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							} else {
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &roldc, result.GetPtr(2))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							Dppt05(uplo, &n, nrhs, asav, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)+1-1), result.Off(3))
						} else {
							k1 = 6
						}

						//                    Compare RCOND from DPPSVX with the computed value
						//                    in RCONDC.
						result.Set(5, Dget06(&rcond, &rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = k1; k <= 6; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								if prefac {
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "DPPSVX", fact, uplo, n, equed, imat, k, result.Get(k-1))
								} else {
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, _type %1d, test(%1d)=%12.5f\n", "DPPSVX", fact, uplo, n, imat, k, result.Get(k-1))
								}
								nfail = nfail + 1
							}
						}
						nrun = nrun + 7 - k1
					label90:
						;
					label100:
					}
				}
			label120:
			}
		label130:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1910
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
