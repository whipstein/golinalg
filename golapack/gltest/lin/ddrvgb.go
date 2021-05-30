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

// Ddrvgb tests the driver routines DGBSV and -SVX.
func Ddrvgb(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, a *mat.Vector, la *int, afb *mat.Vector, lafb *int, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, trans, _type, xtype byte
	var ainvnm, amax, anorm, anormi, anormo, anrmpv, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, i1, i2, iequed, ifact, ikl, iku, imat, in, info, ioff, itran, izero, j, k, k1, kl, ku, lda, ldafb, ldb, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nkl, nku, nrun, nt, ntests, ntran, ntypes int

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
	ntypes = 8
	ntests = 7
	ntran = 3

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	transs[0], transs[1], transs[2] = 'N', 'T', 'C'
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := []byte("DGB")
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
	Xlaenv(2, nbmin)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		ldb = maxint(n, 1)
		xtype = 'N'

		//        Set limits on the number of loop iterations.
		nkl = maxint(1, minint(n, 4))
		if n == 0 {
			nkl = 1
		}
		nku = nkl
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for ikl = 1; ikl <= nkl; ikl++ {
			//           Do for KL = 0, N-1, (3N-1)/4, and (N+1)/4. This order makes
			//           it easier to skip redundant values for small values of N.
			if ikl == 1 {
				kl = 0
			} else if ikl == 2 {
				kl = maxint(n-1, 0)
			} else if ikl == 3 {
				kl = (3*n - 1) / 4
			} else if ikl == 4 {
				kl = (n + 1) / 4
			}
			for iku = 1; iku <= nku; iku++ {
				//              Do for KU = 0, N-1, (3N-1)/4, and (N+1)/4. This order
				//              makes it easier to skip redundant values for small
				//              values of N.
				if iku == 1 {
					ku = 0
				} else if iku == 2 {
					ku = maxint(n-1, 0)
				} else if iku == 3 {
					ku = (3*n - 1) / 4
				} else if iku == 4 {
					ku = (n + 1) / 4
				}
				//              Check that A and AFB are big enough to generate this
				//              matrix.
				lda = kl + ku + 1
				ldafb = 2*kl + ku + 1
				if lda*n > (*la) || ldafb*n > (*lafb) {
					if nfail == 0 && nerrs == 0 {
						Aladhd(path)
					}
					t.Fail()
					if lda*n > (*la) {
						fmt.Printf(" *** In DDRVGB, LA=%5d is too small for N=%5d, KU=%5d, KL=%5d\n ==> Increase LA to at least %5d\n", *la, n, kl, ku, n*(kl+ku+1))
						nerrs = nerrs + 1
					}
					if ldafb*n > (*lafb) {
						fmt.Printf(" *** In DDRVGB, LAFB=%5d is too small for N=%5d, KU=%5d, KL=%5d\n ==> Increase LAFB to at least %5d\n", *lafb, n, kl, ku, n*(2*kl+ku+1))
						nerrs = nerrs + 1
					}
					goto label130
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !(*dotype)[imat-1] {
						goto label120
					}

					//                 Skip types 2, 3, or 4 if the matrix is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label120
					}

					//                 Set up parameters with DLATB4 and generate a
					//                 test matrix with DLATMS.
					Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
					rcondc = one / cndnum
					//
					*srnamt = "DLATMS"
					matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'Z', a.Matrix(lda, opts), &lda, work, &info)
					//
					//                 Check the error code from DLATMS.
					//
					if info != 0 {
						Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
						goto label120
					}

					//                 For types 2, 3, and 4, zero one or more columns of
					//                 the matrix to test that INFO is returned correctly.
					izero = 0
					if zerot {
						if imat == 2 {
							izero = 1
						} else if imat == 3 {
							izero = n
						} else {
							izero = n/2 + 1
						}
						ioff = (izero - 1) * lda
						if imat < 4 {
							i1 = maxint(1, ku+2-izero)
							i2 = minint(kl+ku+1, ku+1+(n-izero))
							for i = i1; i <= i2; i++ {
								a.Set(ioff+i-1, zero)
							}
						} else {
							for j = izero; j <= n; j++ {
								for i = maxint(1, ku+2-j); i <= minint(kl+ku+1, ku+1+(n-j)); i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						}
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Dlacpy('F', toPtr(kl+ku+1), &n, a.Matrix(lda, opts), &lda, asav.Matrix(lda, opts), &lda)

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
									goto label100
								}
								rcondo = zero
								rcondi = zero

							} else if !nofact {
								//                          Compute the condition number for comparison
								//                          with the value returned by DGESVX (FACT =
								//                          'N' reuses the condition number from the
								//                          previous iteration with FACT = 'F').
								golapack.Dlacpy('F', toPtr(kl+ku+1), &n, asav.Matrix(lda, opts), &lda, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb)
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									golapack.Dgbequ(&n, &n, &kl, &ku, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &info)
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

										//                                Equilibrate the matrix.
										golapack.Dlaqgb(&n, &n, &kl, &ku, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in DGET04.
								if equil {
									roldo = rcondo
									roldi = rcondi
								}

								//                          Compute the 1-norm and infinity-norm of A.
								anormo = golapack.Dlangb('1', &n, &kl, &ku, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb, rwork)
								anormi = golapack.Dlangb('I', &n, &kl, &ku, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb, rwork)

								//                          Factor the matrix A.
								golapack.Dgbtrf(&n, &n, &kl, &ku, afb.Matrix(ldafb, opts), &ldafb, iwork, &info)

								//                          Form the inverse of A.
								golapack.Dlaset('F', &n, &n, &zero, &one, work.Matrix(ldb, opts), &ldb)
								*srnamt = "DGBTRS"
								golapack.Dgbtrs('N', &n, &kl, &ku, &n, afb.Matrix(ldafb, opts), &ldafb, iwork, work.Matrix(ldb, opts), &ldb, &info)

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Dlange('1', &n, &n, work.Matrix(ldb, opts), &ldb, rwork)
								if anormo <= zero || ainvnm <= zero {
									rcondo = one
								} else {
									rcondo = (one / anormo) / ainvnm
								}

								//                          Compute the infinity-norm condition number
								//                          of A.
								ainvnm = golapack.Dlange('I', &n, &n, work.Matrix(ldb, opts), &ldb, rwork)
								if anormi <= zero || ainvnm <= zero {
									rcondi = one
								} else {
									rcondi = (one / anormi) / ainvnm
								}
							}

							for itran = 1; itran <= ntran; itran++ {
								//                          Do for each value of TRANS.
								trans = transs[itran-1]
								if itran == 1 {
									rcondc = rcondo
								} else {
									rcondc = rcondi
								}

								//                          Restore the matrix A.
								golapack.Dlacpy('F', toPtr(kl+ku+1), &n, asav.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)

								//                          Form an exact solution and set the right hand
								//                          side.
								*srnamt = "DLARHS"
								Dlarhs(path, &xtype, 'F', trans, &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb, &iseed, &info)
								xtype = 'C'
								golapack.Dlacpy('F', &n, nrhs, b.Matrix(ldb, opts), &ldb, bsav.Matrix(ldb, opts), &ldb)

								if nofact && itran == 1 {
									//                             --- Test DGBSV  ---
									//
									//                             Compute the LU factorization of the matrix
									//                             and solve the system.
									golapack.Dlacpy('F', toPtr(kl+ku+1), &n, a.Matrix(lda, opts), &lda, afb.MatrixOff(kl+1-1, ldafb, opts), &ldafb)
									golapack.Dlacpy('F', &n, nrhs, b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb)

									*srnamt = "DGBSV "
									golapack.Dgbsv(&n, &kl, &ku, nrhs, afb.Matrix(ldafb, opts), &ldafb, iwork, x.Matrix(ldb, opts), &ldb, &info)

									//                             Check error code from DGBSV .
									if info != izero {
										Alaerh(path, []byte("DGBSV "), &info, &izero, []byte(" "), &n, &n, &kl, &ku, nrhs, &imat, &nfail, &nerrs)
									}

									//                             Reconstruct matrix from factors and
									//                             compute residual.
									Dgbt01(&n, &n, &kl, &ku, a.Matrix(lda, opts), &lda, afb.Matrix(ldafb, opts), &ldafb, iwork, work, result.GetPtr(0))
									nt = 1
									if izero == 0 {
										//                                Compute residual of the computed
										//                                solution.
										golapack.Dlacpy('F', &n, nrhs, b.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb)
										Dgbt02('N', &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, x.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb, result.GetPtr(1))

										//                                Check solution from generated exact
										//                                solution.
										Dget04(&n, nrhs, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, &rcondc, result.GetPtr(2))
										nt = 3
									}

									//                             Print information about the tests that did
									//                             not pass the threshold.
									for k = 1; k <= nt; k++ {
										if result.Get(k-1) >= (*thresh) {
											if nfail == 0 && nerrs == 0 {
												Aladhd(path)
											}
											fmt.Printf(" %s, N=%5d, KL=%5d, KU=%5d, _type %1d, test(%1d)=%12.5f\n", "DGBSV ", n, kl, ku, imat, k, result.Get(k-1))
											nfail = nfail + 1
										}
									}
									nrun = nrun + nt
								}

								//                          --- Test DGBSVX ---
								if !prefac {
									golapack.Dlaset('F', toPtr(2*kl+ku+1), &n, &zero, &zero, afb.Matrix(ldafb, opts), &ldafb)
								}
								golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(ldb, opts), &ldb)
								if iequed > 1 && n > 0 {
									//                             Equilibrate the matrix if FACT = 'F' and
									//                             EQUED = 'R', 'C', or 'B'.
									golapack.Dlaqgb(&n, &n, &kl, &ku, a.Matrix(lda, opts), &lda, s, s.Off(n+1-1), &rowcnd, &colcnd, &amax, &equed)
								}

								//                          Solve the system and compute the condition
								//                          number and error bounds using DGBSVX.
								*srnamt = "DGBSVX"
								golapack.Dgbsvx(fact, trans, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, afb.Matrix(ldafb, opts), &ldafb, iwork, &equed, s, s.Off(n+1-1), b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, toSlice(iwork, n+1-1), &info)

								//                          Check the error code from DGBSVX.
								if info != izero {
									Alaerh(path, []byte("DGBSVX"), &info, &izero, []byte{fact, trans}, &n, &n, &kl, &ku, nrhs, &imat, &nfail, &nerrs)
								}

								//                          Compare WORK(1) from DGBSVX with the computed
								//                          reciprocal pivot growth factor RPVGRW
								if info != 0 && info <= n {
									anrmpv = zero
									for j = 1; j <= info; j++ {
										for i = maxint(ku+2-j, 1); i <= minint(n+ku+1-j, kl+ku+1); i++ {
											anrmpv = maxf64(anrmpv, math.Abs(a.Get(i-1+(j-1)*lda)))
										}
									}
									rpvgrw = golapack.Dlantb('M', 'U', 'N', &info, toPtr(minint(info-1, kl+ku)), afb.MatrixOff(maxint(1, kl+ku+2-info)-1, ldafb, opts), &ldafb, work)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = anrmpv / rpvgrw
									}
								} else {
									rpvgrw = golapack.Dlantb('M', 'U', 'N', &n, toPtr(kl+ku), afb.Matrix(ldafb, opts), &ldafb, work)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = golapack.Dlangb('M', &n, &kl, &ku, a.Matrix(lda, opts), &lda, work) / rpvgrw
									}
								}
								result.Set(6, math.Abs(rpvgrw-work.Get(0))/maxf64(work.Get(0), rpvgrw)/golapack.Dlamch(Epsilon))

								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									Dgbt01(&n, &n, &kl, &ku, a.Matrix(lda, opts), &lda, afb.Matrix(ldafb, opts), &ldafb, iwork, work, result.GetPtr(0))
									k1 = 1
								} else {
									k1 = 2
								}

								if info == 0 {
									trfcon = false

									//                             Compute residual of the computed solution.
									golapack.Dlacpy('F', &n, nrhs, bsav.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb)
									Dgbt02(trans, &n, &n, &kl, &ku, nrhs, asav.Matrix(lda, opts), &lda, x.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb, result.GetPtr(1))

									//                             Check solution from generated exact
									//                             solution.
									if nofact || (prefac && equed == 'N') {
										Dget04(&n, nrhs, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, &rcondc, result.GetPtr(2))
									} else {
										if itran == 1 {
											roldc = roldo
										} else {
											roldc = roldi
										}
										Dget04(&n, nrhs, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, &roldc, result.GetPtr(2))
									}

									//                             Check the error bounds from iterative
									//                             refinement.
									Dgbt05(trans, &n, &kl, &ku, nrhs, asav.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, rwork, rwork.Off((*nrhs)+1-1), result.Off(3))
								} else {
									trfcon = true
								}

								//                          Compare RCOND from DGBSVX with the computed
								//                          value in RCONDC.
								result.Set(5, Dget06(&rcond, &rcondc))

								//                          Print information about the tests that did
								//                          not pass the threshold.
								if !trfcon {
									for k = k1; k <= ntests; k++ {
										if result.Get(k-1) >= (*thresh) {
											if nfail == 0 && nerrs == 0 {
												Aladhd(path)
											}
											t.Fail()
											if prefac {
												fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, equed, imat, k, result.Get(k-1))
											} else {
												fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, imat, k, result.Get(k-1))
											}
											nfail = nfail + 1
										}
									}
									nrun = nrun + ntests - k1 + 1
								} else {
									if result.Get(0) >= (*thresh) && !prefac {
										if nfail == 0 && nerrs == 0 {
											Aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, equed, imat, 1, result.Get(0))
										} else {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, imat, 1, result.Get(0))
										}
										nfail = nfail + 1
										nrun = nrun + 1
									}
									if result.Get(5) >= (*thresh) {
										if nfail == 0 && nerrs == 0 {
											Aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, equed, imat, 6, result.Get(5))
										} else {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, imat, 6, result.Get(5))
										}
										nfail = nfail + 1
										nrun = nrun + 1
									}
									if result.Get(6) >= (*thresh) {
										if nfail == 0 && nerrs == 0 {
											Aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, equed, imat, 7, result.Get(6))
										} else {
											fmt.Printf(" %s( '%c','%c',%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "DGBSVX", fact, trans, n, kl, ku, imat, 7, result.Get(6))
										}
										nfail = nfail + 1
										nrun = nrun + 1
									}

								}
							}
						label100:
						}
					}
				label120:
				}
			label130:
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 36567
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
