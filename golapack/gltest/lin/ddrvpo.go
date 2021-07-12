package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Ddrvpo tests the driver routines DPOSV and -SVX.
func Ddrvpo(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var equil, nofact, prefac, zerot bool
	var dist, equed, fact, _type, uplo, xtype byte
	var ainvnm, amax, anorm, cndnum, one, rcond, rcondc, roldc, scond, zero float64
	var i, iequed, ifact, imat, in, info, ioff, iuplo, izero, k, k1, kl, ku, lda, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntypes int

	equeds := make([]byte, 2)
	facts := make([]byte, 3)
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
	uplos[0], uplos[1] = 'U', 'L'
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1] = 'N', 'Y'

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
		Derrvx(path, t)
	}
	(*infot) = 0

	//     Set the block size and minimum block size for testing.
	nb = 1
	nbmin = 2
	Xlaenv(1, nb)
	Xlaenv(1, nbmin)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = max(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label120
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label120
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
					goto label110
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
					//
					//                 Set row and column IZERO of A to 0.
					//
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

				//              Save a copy of the matrix A in ASAV.
				golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, asav.Matrix(lda, opts), &lda)

				for iequed = 1; iequed <= 2; iequed++ {
					equed = equeds[iequed-1]
					if iequed == 1 {
						nfact = 3
					} else {
						nfact = 1
					}

					for ifact = 1; ifact <= nfact; ifact++ {
						fact = facts[ifact-1]
						prefac = fact == 'F'
						nofact = fact == 'N'
						equil = fact == 'E'

						if zerot {
							if prefac {
								goto label90
							}
							rcondc = zero

						} else if fact != 'N' {
							//                       Compute the condition number for comparison with
							//                       the value returned by DPOSVX (FACT = 'N' reuses
							//                       the condition number from the previous iteration
							//                       with FACT = 'F').
							golapack.Dlacpy(uplo, &n, &n, asav.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
							if equil || iequed > 1 {
								//                          Compute row and column scale factors to
								//                          equilibrate the matrix A.
								golapack.Dpoequ(&n, afac.Matrix(lda, opts), &lda, s, &scond, &amax, &info)
								if info == 0 && n > 0 {
									if iequed > 1 {
										scond = zero
									}

									//                             Equilibrate the matrix.
									golapack.Dlaqsy(uplo, &n, afac.Matrix(lda, opts), &lda, s, &scond, &amax, &equed)
								}
							}

							//                       Save the condition number of the
							//                       non-equilibrated system for use in DGET04.
							if equil {
								roldc = rcondc
							}

							//                       Compute the 1-norm of A.
							anorm = golapack.Dlansy('1', uplo, &n, afac.Matrix(lda, opts), &lda, rwork)

							//                       Factor the matrix A.
							golapack.Dpotrf(uplo, &n, afac.Matrix(lda, opts), &lda, &info)

							//                       Form the inverse of A.
							golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
							golapack.Dpotri(uplo, &n, a.Matrix(lda, opts), &lda, &info)

							//                       Compute the 1-norm condition number of A.
							ainvnm = golapack.Dlansy('1', uplo, &n, a.Matrix(lda, opts), &lda, rwork)
							if anorm <= zero || ainvnm <= zero {
								rcondc = one
							} else {
								rcondc = (one / anorm) / ainvnm
							}
						}

						//                    Restore the matrix A.
						golapack.Dlacpy(uplo, &n, &n, asav.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)

						//                    Form an exact solution and set the right hand side.
						*srnamt = "DLARHS"
						Dlarhs(path, &xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, bsav.Matrix(lda, opts), &lda)

						if nofact {
							//                       --- Test DPOSV  ---
							//
							//                       Compute the L*L' or U'*U factorization of the
							//                       matrix and solve the system.
							golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

							*srnamt = "DPOSV "
							golapack.Dposv(uplo, &n, nrhs, afac.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &info)

							//                       Check error code from DPOSV .
							if info != izero {
								Alaerh(path, []byte("DPOSV "), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
								goto label70
							} else if info != 0 {
								goto label70
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dpot01(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))

							//                       Compute residual of the computed solution.
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dpot02(uplo, &n, nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

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
									fmt.Printf(" %s, UPLO='%c', N =%5d, _type %1d, test(%1d)=%12.5f\n", "DPOSV ", uplo, n, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + nt
						label70:
						}

						//                    --- Test DPOSVX ---
						if !prefac {
							golapack.Dlaset(uplo, &n, &n, &zero, &zero, afac.Matrix(lda, opts), &lda)
						}
						golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(lda, opts), &lda)
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT='F' and
							//                       EQUED='Y'.
							golapack.Dlaqsy(uplo, &n, a.Matrix(lda, opts), &lda, s, &scond, &amax, &equed)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using DPOSVX.
						*srnamt = "DPOSVX"
						golapack.Dposvx(fact, uplo, &n, nrhs, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, &equed, s, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)), work, iwork, &info)

						//                    Check the error code from DPOSVX.
						if info != izero {
							Alaerh(path, []byte("DPOSVX"), &info, &izero, []byte{fact, uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label90
						}

						if info == 0 {
							if !prefac {
								//                          Reconstruct matrix from factors and compute
								//                          residual.
								Dpot01(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)), result.GetPtr(0))
								k1 = 1
							} else {
								k1 = 2
							}

							//                       Compute residual of the computed solution.
							golapack.Dlacpy('F', &n, nrhs, bsav.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dpot02(uplo, &n, nrhs, asav.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)), result.GetPtr(1))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							} else {
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &roldc, result.GetPtr(2))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							Dpot05(uplo, &n, nrhs, asav.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)), result.Off(3))
						} else {
							k1 = 6
						}

						//                    Compare RCOND from DPOSVX with the computed value
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
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, EQUED='%c', _type %1d, test(%1d) =%12.5f\n", "DPOSVX", fact, uplo, n, equed, imat, k, result.Get(k-1))
								} else {
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, _type %1d, test(%1d)=%12.5f\n", "DPOSVX", fact, uplo, n, imat, k, result.Get(k-1))
								}
								nfail = nfail + 1
							}
						}
						nrun = nrun + 7 - k1
					label90:
					}
				}
			label110:
			}
		label120:
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
