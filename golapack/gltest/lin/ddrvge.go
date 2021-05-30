package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Ddrvge tests the driver routines DGESV and -SVX.
func Ddrvge(dotype []bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr bool, nmax *int, a, afac, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, trans, _type, xtype byte
	var ainvnm, amax, anorm, anormi, anormo, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, iequed, ifact, imat, in, info, ioff, itran, izero, k, k1, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntests, ntran, ntypes int

	equeds := make([]byte, 4)
	facts := make([]byte, 3)
	transs := make([]byte, 3)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 11
	ntests = 7
	ntran = 3

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	transs[0], transs[1], transs[2] = 'N', 'T', 'C'
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := []byte("DGE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		Derrvx(path, t)
	}
	(*infot) = 0

	//     Set the block size and minimum block size for testing.
	nb = 1
	nbmin = 2
	Xlaenv(1, nb)
	Xlaenv(2, nbmin)

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
			if !dotype[imat-1] {
				goto label80
			}

			//           Skip types 5, 6, or 7 if the matrix size is too small.
			zerot = imat >= 5 && imat <= 7
			if zerot && n < imat-4 {
				goto label80
			}

			//           Set up parameters with DLATB4 and generate a test matrix
			//           with DLATMS.
			Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
			rcondc = one / cndnum

			*srnamt = "DLATMS"
			matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'N', a.Matrix(lda, opts), &lda, work, &info)

			//           Check error code from DLATMS.
			if info != 0 {
				Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				goto label80
			}

			//           For types 5-7, zero one or more columns of the matrix to
			//           test that INFO is returned correctly.
			if zerot {
				if imat == 5 {
					izero = 1
				} else if imat == 6 {
					izero = n
				} else {
					izero = n/2 + 1
				}
				ioff = (izero - 1) * lda
				if imat < 7 {
					for i = 1; i <= n; i++ {
						a.Set(ioff+i-1, zero)
					}
				} else {
					golapack.Dlaset('F', &n, toPtr(n-izero+1), &zero, &zero, a.MatrixOff(ioff+1-1, lda, opts), &lda)
				}
			} else {
				izero = 0
			}

			//           Save a copy of the matrix A in ASAV.
			golapack.Dlacpy('F', &n, &n, a.Matrix(lda, opts), &lda, asav.Matrix(lda, opts), &lda)

			for iequed = 1; iequed <= 4; iequed++ {
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
							goto label60
						}
						rcondo = zero
						rcondi = zero

					} else if !nofact {
						//                    Compute the condition number for comparison with
						//                    the value returned by DGESVX (FACT = 'N' reuses
						//                    the condition number from the previous iteration
						//                    with FACT = 'F').
						golapack.Dlacpy('F', &n, &n, asav.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
						if equil || iequed > 1 {
							//                       Compute row and column scale factors to
							//                       equilibrate the matrix A.
							golapack.Dgeequ(&n, &n, afac.Matrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &info)
							if info == 0 && n > 0 {
								if equed == 'R' {
									rowcnd = zero
									colcnd = one
								} else if equed == 'C' {
									rowcnd = one
									colcnd = zero
								} else if equed == 'B' {
									rowcnd = zero
									colcnd = zero
								}

								//                          Equilibrate the matrix.
								golapack.Dlaqge(&n, &n, afac.Matrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
							}
						}

						//                    Save the condition number of the non-equilibrated
						//                    system for use in DGET04.
						if equil {
							roldo = rcondo
							roldi = rcondi
						}

						//                    Compute the 1-norm and infinity-norm of A.
						anormo = golapack.Dlange('1', &n, &n, afac.Matrix(lda, opts), &lda, rwork)
						anormi = golapack.Dlange('I', &n, &n, afac.Matrix(lda, opts), &lda, rwork)

						//                    Factor the matrix A.
						*srnamt = "DGETRF"
						golapack.Dgetrf(&n, &n, afac.Matrix(lda, opts), &lda, iwork, &info)

						//                    Form the inverse of A.
						golapack.Dlacpy('F', &n, &n, afac.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
						lwork = (*nmax) * maxint(3, *nrhs)
						*srnamt = "DGETRI"
						golapack.Dgetri(&n, a.Matrix(lda, opts), &lda, iwork, work.Matrix(lda, opts), &lwork, &info)

						//                    Compute the 1-norm condition number of A.
						ainvnm = golapack.Dlange('1', &n, &n, a.Matrix(lda, opts), &lda, rwork)
						if anormo <= zero || ainvnm <= zero {
							rcondo = one
						} else {
							rcondo = (one / anormo) / ainvnm
						}

						//                    Compute the infinity-norm condition number of A.
						ainvnm = golapack.Dlange('I', &n, &n, a.Matrix(lda, opts), &lda, rwork)
						if anormi <= zero || ainvnm <= zero {
							rcondi = one
						} else {
							rcondi = (one / anormi) / ainvnm
						}
					}

					for itran = 1; itran <= ntran; itran++ {
						//                    Do for each value of TRANS.
						trans = transs[itran-1]
						if itran == 1 {
							rcondc = rcondo
						} else {
							rcondc = rcondi
						}

						//                    Restore the matrix A.
						golapack.Dlacpy('F', &n, &n, asav.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)

						//                    Form an exact solution and set the right hand side.
						*srnamt = "DLARHS"
						Dlarhs(path, &xtype, 'F', trans, &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, bsav.Matrix(lda, opts), &lda)

						if nofact && itran == 1 {
							//                       --- Test DGESV  ---
							//
							//                       Compute the LU factorization of the matrix and
							//                       solve the system.
							golapack.Dlacpy('F', &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

							*srnamt = "DGESV "
							golapack.Dgesv(&n, nrhs, afac.Matrix(lda, opts), &lda, iwork, x.Matrix(lda, opts), &lda, &info)

							//                       Check error code from DGESV .
							if info != izero {
								Alaerh(path, []byte("DGESV "), &info, &izero, []byte(" "), &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dget01(&n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, rwork, result.GetPtr(0))
							nt = 1
							if izero == 0 {
								//                          Compute residual of the computed solution.
								golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
								Dget02('N', &n, &n, nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

								//                          Check solution from generated exact solution.
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
								nt = 3
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									t.Fail()
									fmt.Printf(" %s, N =%5d, _type %2d, test(%2d) =%12.5f\n", "DGESV ", n, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += nt
						}

						//                    --- Test DGESVX ---
						if !prefac {
							golapack.Dlaset('F', &n, &n, &zero, &zero, afac.Matrix(lda, opts), &lda)
						}
						golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(lda, opts), &lda)
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT = 'F' and
							//                       EQUED = 'R', 'C', or 'B'.
							golapack.Dlaqge(&n, &n, a.Matrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using DGESVX.
						*srnamt = "DGESVX"
						golapack.Dgesvx(fact, trans, &n, nrhs, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, &equed, s, s.Off(n+1-1), b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, toSlice(iwork, n+1-1), &info)

						//                    Check the error code from DGESVX.
						if info != izero {
							Alaerh(path, []byte("DGESVX"), &info, &izero, []byte{fact, trans}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
						}

						//                    Compare WORK(1) from DGESVX with the computed
						//                    reciprocal pivot growth factor RPVGRW
						if info != 0 && info <= n {
							rpvgrw = golapack.Dlantr('M', 'U', 'N', &info, &info, afac.Matrix(lda, opts), &lda, work)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Dlange('M', &n, &info, a.Matrix(lda, opts), &lda, work) / rpvgrw
							}
						} else {
							rpvgrw = golapack.Dlantr('M', 'U', 'N', &n, &n, afac.Matrix(lda, opts), &lda, work)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Dlange('M', &n, &n, a.Matrix(lda, opts), &lda, work) / rpvgrw
							}
						}
						result.Set(6, math.Abs(rpvgrw-work.Get(0))/maxf64(work.Get(0), rpvgrw)/golapack.Dlamch(Epsilon))

						if !prefac {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dget01(&n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(0))
							k1 = 1
						} else {
							k1 = 2
						}

						if info == 0 {
							trfcon = false

							//                       Compute residual of the computed solution.
							golapack.Dlacpy('F', &n, nrhs, bsav.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dget02(trans, &n, &n, nrhs, asav.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(1))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							} else {
								if itran == 1 {
									roldc = roldo
								} else {
									roldc = roldi
								}
								Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &roldc, result.GetPtr(2))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							Dget07(trans, &n, nrhs, asav.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, true, rwork.Off((*nrhs)+1-1), result.Off(3))
						} else {
							trfcon = true
						}

						//                    Compare RCOND from DGESVX with the computed value
						//                    in RCONDC.
						result.Set(5, Dget06(&rcond, &rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if !trfcon {
							for k = k1; k <= ntests; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									t.Fail()
									if prefac {
										fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, imat, k, result.Get(k-1))
									}
									nfail++
								}
							}
							nrun += ntests - k1 + 1
						} else {
							if result.Get(0) >= (*thresh) && !prefac {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, equed, imat, 1, result.Get(0))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, imat, 1, result.Get(0))
								}
								nfail++
								nrun++
							}
							if result.Get(5) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, equed, imat, 6, result.Get(5))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, imat, 6, result.Get(5))
								}
								nfail++
								nrun++
							}
							if result.Get(6) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, equed, imat, 7, result.Get(6))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "DGESVX", fact, trans, n, imat, 7, result.Get(6))
								}
								nfail++
								nrun++
							}

						}

					}
				label60:
				}
			}
		label80:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 5748
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
