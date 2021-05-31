package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zdrvge tests the driver routines ZGESV and -SVX.
func Zdrvge(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, trans, _type, xtype byte
	var ainvnm, amax, anorm, anormi, anormo, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, iequed, ifact, imat, in, info, ioff, itran, izero, k, k1, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntests, ntran, ntypes int

	equeds := make([]byte, 4)
	facts := make([]byte, 3)
	transs := make([]byte, 3)
	rdum := vf(1)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 11
	ntests = 7
	ntran = 3

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	transs[0], transs[1], transs[2] = 'N', 'T', 'C'
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := []byte("ZGE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrvx(path, t)
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
			if !(*dotype)[imat-1] {
				goto label80
			}

			//           Skip types 5, 6, or 7 if the matrix size is too small.
			zerot = imat >= 5 && imat <= 7
			if zerot && n < imat-4 {
				goto label80
			}

			//           Set up parameters with ZLATB4 and generate a test matrix
			//           with ZLATMS.
			Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
			rcondc = one / cndnum

			*srnamt = "ZLATMS"
			matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'N', a.CMatrix(lda, opts), &lda, work, &info)

			//           Check error code from ZLATMS.
			if info != 0 {
				t.Fail()
				Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
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
						a.SetRe(ioff+i-1, zero)
					}
				} else {
					golapack.Zlaset('F', &n, toPtr(n-izero+1), toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), a.CMatrixOff(ioff+1-1, lda, opts), &lda)
				}
			} else {
				izero = 0
			}

			//           Save a copy of the matrix A in ASAV.
			golapack.Zlacpy('F', &n, &n, a.CMatrix(lda, opts), &lda, asav.CMatrix(lda, opts), &lda)

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
						//                    the value returned by ZGESVX (FACT = 'N' reuses
						//                    the condition number from the previous iteration
						//                    with FACT = 'F').
						golapack.Zlacpy('F', &n, &n, asav.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)
						if equil || iequed > 1 {
							//                       Compute row and column scale factors to
							//                       equilibrate the matrix A.
							golapack.Zgeequ(&n, &n, afac.CMatrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &info)
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
								golapack.Zlaqge(&n, &n, afac.CMatrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
							}
						}

						//                    Save the condition number of the non-equilibrated
						//                    system for use in ZGET04.
						if equil {
							roldo = rcondo
							roldi = rcondi
						}

						//                    Compute the 1-norm and infinity-norm of A.
						anormo = golapack.Zlange('1', &n, &n, afac.CMatrix(lda, opts), &lda, rwork)
						anormi = golapack.Zlange('I', &n, &n, afac.CMatrix(lda, opts), &lda, rwork)

						//                    Factor the matrix A.
						*srnamt = "ZGETRF"
						golapack.Zgetrf(&n, &n, afac.CMatrix(lda, opts), &lda, iwork, &info)

						//                    Form the inverse of A.
						golapack.Zlacpy('F', &n, &n, afac.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
						lwork = (*nmax) * maxint(3, *nrhs)
						*srnamt = "ZGETRI"
						golapack.Zgetri(&n, a.CMatrix(lda, opts), &lda, iwork, work, &lwork, &info)

						//                    Compute the 1-norm condition number of A.
						ainvnm = golapack.Zlange('1', &n, &n, a.CMatrix(lda, opts), &lda, rwork)
						if anormo <= zero || ainvnm <= zero {
							rcondo = one
						} else {
							rcondo = (one / anormo) / ainvnm
						}

						//                    Compute the infinity-norm condition number of A.
						ainvnm = golapack.Zlange('I', &n, &n, a.CMatrix(lda, opts), &lda, rwork)
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
						golapack.Zlacpy('F', &n, &n, asav.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)

						//                    Form an exact solution and set the right hand side.
						*srnamt = "ZLARHS"
						Zlarhs(path, xtype, 'F', trans, &n, &n, &kl, &ku, nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, bsav.CMatrix(lda, opts), &lda)

						if nofact && itran == 1 {
							//                       --- Test ZGESV  ---
							//
							//                       Compute the LU factorization of the matrix and
							//                       solve the system.
							golapack.Zlacpy('F', &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)
							golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

							*srnamt = "ZGESV "
							golapack.Zgesv(&n, nrhs, afac.CMatrix(lda, opts), &lda, iwork, x.CMatrix(lda, opts), &lda, &info)

							//                       Check error code from ZGESV .
							if info != izero {
								t.Fail()
								Alaerh(path, []byte("ZGESV "), &info, &izero, []byte{' '}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Zget01(&n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, rwork, result.GetPtr(0))
							nt = 1
							if izero == 0 {
								//                          Compute residual of the computed solution.
								golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
								Zget02('N', &n, &n, nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

								//                          Check solution from generated exact solution.
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
								nt = 3
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									fmt.Printf(" %s, N =%5d, _type %2d, test(%2d) =%12.5f\n", "ZGESV ", n, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + nt
						}

						//                    --- Test ZGESVX ---
						if !prefac {
							golapack.Zlaset('F', &n, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), afac.CMatrix(lda, opts), &lda)
						}
						golapack.Zlaset('F', &n, nrhs, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), x.CMatrix(lda, opts), &lda)
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT = 'F' and
							//                       EQUED = 'R', 'C', or 'B'.
							golapack.Zlaqge(&n, &n, a.CMatrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using ZGESVX.
						*srnamt = "ZGESVX"
						golapack.Zgesvx(fact, trans, &n, nrhs, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, &equed, s, s.Off(n+1-1), b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, rwork.Off(2*(*nrhs)+1-1), &info)

						//                    Check the error code from ZGESVX.
						if info != izero {
							t.Fail()
							Alaerh(path, []byte("ZGESVX"), &info, &izero, append(append([]byte{}, fact), trans), &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
						}

						//                    Compare RWORK(2*NRHS+1) from ZGESVX with the
						//                    computed reciprocal pivot growth factor RPVGRW
						if info != 0 && info <= n {
							rpvgrw = golapack.Zlantr('M', 'U', 'N', &info, &info, afac.CMatrix(lda, opts), &lda, rdum)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Zlange('M', &n, &info, a.CMatrix(lda, opts), &lda, rdum) / rpvgrw
							}
						} else {
							rpvgrw = golapack.Zlantr('M', 'U', 'N', &n, &n, afac.CMatrix(lda, opts), &lda, rdum)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Zlange('M', &n, &n, a.CMatrix(lda, opts), &lda, rdum) / rpvgrw
							}
						}
						result.Set(6, math.Abs(rpvgrw-rwork.Get(2*(*nrhs)+1-1))/maxf64(rwork.Get(2*(*nrhs)+1-1), rpvgrw)/golapack.Dlamch(Epsilon))

						if !prefac {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Zget01(&n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(0))
							k1 = 1
						} else {
							k1 = 2
						}

						if info == 0 {
							trfcon = false

							//                       Compute residual of the computed solution.
							golapack.Zlacpy('F', &n, nrhs, bsav.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
							Zget02(trans, &n, &n, nrhs, asav.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(1))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							} else {
								if itran == 1 {
									roldc = roldo
								} else {
									roldc = roldi
								}
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &roldc, result.GetPtr(2))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							Zget07(trans, &n, nrhs, asav.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, true, rwork.Off((*nrhs)+1-1), result.Off(3))
						} else {
							trfcon = true
						}

						//                    Compare RCOND from ZGESVX with the computed value
						//                    in RCONDC.
						result.Set(5, Dget06(&rcond, &rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if !trfcon {
							for k = k1; k <= ntests; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									if prefac {
										fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, imat, k, result.Get(k-1))
									}
									nfail = nfail + 1
								}
							}
							nrun = nrun + ntests - k1 + 1
						} else {
							if result.Get(0) >= (*thresh) && !prefac {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, equed, imat, 1, result.Get(0))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, imat, 1, result.Get(0))
								}
								nfail = nfail + 1
								nrun = nrun + 1
							}
							if result.Get(5) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, equed, imat, 6, result.Get(5))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, imat, 6, result.Get(5))
								}
								nfail = nfail + 1
								nrun = nrun + 1
							}
							if result.Get(6) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, EQUED='%c', _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, equed, imat, 7, result.Get(6))
								} else {
									fmt.Printf(" %s, FACT='%c', TRANS='%c', N=%5d, _type %2d, test(%1d)=%12.5f\n", "ZGESVX", fact, trans, n, imat, 7, result.Get(6))
								}
								nfail = nfail + 1
								nrun = nrun + 1
							}

						}

					}
				label60:
				}
			}
		label80:
		}
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
