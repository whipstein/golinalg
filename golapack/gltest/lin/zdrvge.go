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

// zdrvge tests the driver routines Zgesvand -SVX.
func zdrvge(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, afac, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, _type, xtype byte
	var trans mat.MatTrans
	var ainvnm, amax, anorm, anormi, anormo, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, iequed, ifact, imat, in, info, ioff, izero, k, k1, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntests, ntypes int
	var err error

	equeds := make([]byte, 4)
	facts := make([]byte, 3)
	rdum := vf(1)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 11
	ntests = 7

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := "Zge"
	alasvmStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrvx(path, t)
	}
	(*infot) = 0

	//     Set the block size and minimum block size for testing.
	nb = 1
	nbmin = 2
	xlaenv(1, nb)
	xlaenv(2, nbmin)

	//     Do for each value of N in NVAL
	for in = 1; in <= nn; in++ {
		n = nval[in-1]
		lda = max(n, 1)
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

			//           Set up parameters with ZLATB4 and generate a test matrix
			//           with Zlatms.
			_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)
			rcondc = one / cndnum

			*srnamt = "Zlatms"
			if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a.CMatrix(lda, opts), work); err != nil {
				t.Fail()
				nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, n, n, -1, -1, -1, imat, nfail, nerrs)
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
					golapack.Zlaset(Full, n, n-izero+1, complex(zero, 0), complex(zero, 0), a.Off(ioff).CMatrix(lda, opts))
				}
			} else {
				izero = 0
			}

			//           Save a copy of the matrix A in ASAV.
			golapack.Zlacpy(Full, n, n, a.CMatrix(lda, opts), asav.CMatrix(lda, opts))

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
						//                    the value returned by Zgesvx (fact = 'N' reuses
						//                    the condition number from the previous iteration
						//                    with fact = 'F').
						golapack.Zlacpy(Full, n, n, asav.CMatrix(lda, opts), afac.CMatrix(lda, opts))
						if equil || iequed > 1 {
							//                       Compute row and column scale factors to
							//                       equilibrate the matrix A.
							if rowcnd, colcnd, amax, info, err = golapack.Zgeequ(n, n, afac.CMatrix(lda, opts), s, s.Off(n)); err != nil {
								panic(err)
							}
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
								equed = golapack.Zlaqge(n, n, afac.CMatrix(lda, opts), s, s.Off(n), rowcnd, colcnd, amax)
							}
						}

						//                    Save the condition number of the non-equilibrated
						//                    system for use in ZGET04.
						if equil {
							roldo = rcondo
							roldi = rcondi
						}

						//                    Compute the 1-norm and infinity-norm of A.
						anormo = golapack.Zlange('1', n, n, afac.CMatrix(lda, opts), rwork)
						anormi = golapack.Zlange('I', n, n, afac.CMatrix(lda, opts), rwork)

						//                    Factor the matrix A.
						*srnamt = "Zgetrf"
						if info, err = golapack.Zgetrf(n, n, afac.CMatrix(lda, opts), &iwork); err != nil {
							panic(err)
						}

						//                    Form the inverse of A.
						golapack.Zlacpy(Full, n, n, afac.CMatrix(lda, opts), a.CMatrix(lda, opts))
						lwork = nmax * max(3, nrhs)
						*srnamt = "Zgetri"
						if info, err = golapack.Zgetri(n, a.CMatrix(lda, opts), &iwork, work, lwork); err != nil {
							panic(err)
						}

						//                    Compute the 1-norm condition number of A.
						ainvnm = golapack.Zlange('1', n, n, a.CMatrix(lda, opts), rwork)
						if anormo <= zero || ainvnm <= zero {
							rcondo = one
						} else {
							rcondo = (one / anormo) / ainvnm
						}

						//                    Compute the infinity-norm condition number of A.
						ainvnm = golapack.Zlange('I', n, n, a.CMatrix(lda, opts), rwork)
						if anormi <= zero || ainvnm <= zero {
							rcondi = one
						} else {
							rcondi = (one / anormi) / ainvnm
						}
					}

					for _, trans = range mat.IterMatTrans() {
						//                    Do for each value of trans.
						if trans == NoTrans {
							rcondc = rcondo
						} else {
							rcondc = rcondi
						}

						//                    Restore the matrix A.
						golapack.Zlacpy(Full, n, n, asav.CMatrix(lda, opts), a.CMatrix(lda, opts))

						//                    Form an exact solution and set the right hand side.
						*srnamt = "zlarhs"
						if err = zlarhs(path, xtype, Full, trans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						xtype = 'C'
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), bsav.CMatrix(lda, opts))

						if nofact && trans == NoTrans {
							//                       --- Test Zgesv ---
							//
							//                       Compute the LU factorization of the matrix and
							//                       solve the system.
							golapack.Zlacpy(Full, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

							*srnamt = "Zgesv"
							if info, err = golapack.Zgesv(n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts)); err != nil {
								panic(err)
							}

							//                       Check error code from Zgesv.
							if info != izero {
								t.Fail()
								nerrs = alaerh(path, "Zgesv", info, 0, []byte{' '}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							*result.GetPtr(0) = zget01(n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, rwork)
							nt = 1
							if izero == 0 {
								//                          Compute residual of the computed solution.
								golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
								*result.GetPtr(1) = zget02(NoTrans, n, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

								//                          Check solution from generated exact solution.
								*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
								nt = 3
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										aladhd(path)
									}
									fmt.Printf(" %s, n=%5d, _type %2d, test(%2d) =%12.5f\n", "Zgesv", n, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun = nrun + nt
						}

						//                    --- Test Zgesvx ---
						if !prefac {
							golapack.Zlaset(Full, n, n, complex(zero, 0), complex(zero, 0), afac.CMatrix(lda, opts))
						}
						golapack.Zlaset(Full, n, nrhs, complex(zero, 0), complex(zero, 0), x.CMatrix(lda, opts))
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if fact = 'F' and
							//                       equed = 'R', 'C', or 'B'.
							equed = golapack.Zlaqge(n, n, a.CMatrix(lda, opts), s, s.Off(n), rowcnd, colcnd, amax)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using Zgesvx.
						*srnamt = "Zgesvx"
						if equed, rcond, info, err = golapack.Zgesvx(fact, trans, n, nrhs, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, equed, s, s.Off(n), b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
							panic(err)
						}

						//                    Check the error code from Zgesvx.
						if info != izero {
							t.Fail()
							nerrs = alaerh(path, "Zgesvx", info, 0, []byte{fact, trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						//                    Compare RWORK(2*nrhs+1) from Zgesvx with the
						//                    computed reciprocal pivot growth factor RPVGRW
						if info != 0 && info <= n {
							rpvgrw = golapack.Zlantr('M', Upper, NonUnit, info, info, afac.CMatrix(lda, opts), rdum)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Zlange('M', n, info, a.CMatrix(lda, opts), rdum) / rpvgrw
							}
						} else {
							rpvgrw = golapack.Zlantr('M', Upper, NonUnit, n, n, afac.CMatrix(lda, opts), rdum)
							if rpvgrw == zero {
								rpvgrw = one
							} else {
								rpvgrw = golapack.Zlange('M', n, n, a.CMatrix(lda, opts), rdum) / rpvgrw
							}
						}
						result.Set(6, math.Abs(rpvgrw-rwork.Get(2*nrhs))/math.Max(rwork.Get(2*nrhs), rpvgrw)/golapack.Dlamch(Epsilon))

						if !prefac {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							*result.GetPtr(0) = zget01(n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, rwork.Off(2*nrhs))
							k1 = 1
						} else {
							k1 = 2
						}

						if info == 0 {
							trfcon = false

							//                       Compute residual of the computed solution.
							golapack.Zlacpy(Full, n, nrhs, bsav.CMatrix(lda, opts), work.CMatrix(lda, opts))
							*result.GetPtr(1) = zget02(trans, n, n, nrhs, asav.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork.Off(2*nrhs))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							} else {
								if trans == NoTrans {
									roldc = roldo
								} else {
									roldc = roldi
								}
								*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), roldc)
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							zget07(trans, n, nrhs, asav.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, true, rwork.Off(nrhs), result.Off(3))
						} else {
							trfcon = true
						}

						//                    Compare RCOND from Zgesvx with the computed value
						//                    in RCONDC.
						result.Set(5, dget06(rcond, rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if !trfcon {
							for k = k1; k <= ntests; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										aladhd(path)
									}
									if prefac {
										fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, equed='%c', _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, imat, k, result.Get(k-1))
									}
									nfail++
								}
							}
							nrun = nrun + ntests - k1 + 1
						} else {
							if result.Get(0) >= thresh && !prefac {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, equed='%c', _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, equed, imat, 1, result.Get(0))
								} else {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, imat, 1, result.Get(0))
								}
								nfail++
								nrun++
							}
							if result.Get(5) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, equed='%c', _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, equed, imat, 6, result.Get(5))
								} else {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, imat, 6, result.Get(5))
								}
								nfail++
								nrun++
							}
							if result.Get(6) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, equed='%c', _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, equed, imat, 7, result.Get(6))
								} else {
									fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test(%1d)=%12.5f\n", "Zgesvx", fact, trans, n, imat, 7, result.Get(6))
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

	//     Print a summary of the results.
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
