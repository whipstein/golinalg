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

// ddrvgb tests the driver routines Dgbsvand -SVX.
func ddrvgb(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, a *mat.Vector, la int, afb *mat.Vector, lafb int, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, _type, xtype byte
	var trans mat.MatTrans
	var ainvnm, amax, anorm, anormi, anormo, anrmpv, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, i1, i2, iequed, ifact, ikl, iku, imat, in, info, ioff, izero, j, k, k1, kl, ku, lda, ldafb, ldb, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nkl, nku, nrun, nt, ntests, ntypes int
	var err error

	equeds := make([]byte, 4)
	facts := make([]byte, 3)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 8
	ntests = 7

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := "Dgb"
	alasvmStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrvx(path, t)
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
		ldb = max(n, 1)
		xtype = 'N'

		//        Set limits on the number of loop iterations.
		nkl = max(1, min(n, 4))
		if n == 0 {
			nkl = 1
		}
		nku = nkl
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for ikl = 1; ikl <= nkl; ikl++ {
			//           Do for kl = 0, N-1, (3N-1)/4, and (N+1)/4. This order makes
			//           it easier to skip redundant values for small values of N.
			if ikl == 1 {
				kl = 0
			} else if ikl == 2 {
				kl = max(n-1, 0)
			} else if ikl == 3 {
				kl = (3*n - 1) / 4
			} else if ikl == 4 {
				kl = (n + 1) / 4
			}
			for iku = 1; iku <= nku; iku++ {
				//              Do for ku = 0, N-1, (3N-1)/4, and (N+1)/4. This order
				//              makes it easier to skip redundant values for small
				//              values of N.
				if iku == 1 {
					ku = 0
				} else if iku == 2 {
					ku = max(n-1, 0)
				} else if iku == 3 {
					ku = (3*n - 1) / 4
				} else if iku == 4 {
					ku = (n + 1) / 4
				}
				//              Check that A and AFB are big enough to generate this
				//              matrix.
				lda = kl + ku + 1
				ldafb = 2*kl + ku + 1
				if lda*n > la || ldafb*n > lafb {
					if nfail == 0 && nerrs == 0 {
						aladhd(path)
					}
					t.Fail()
					if lda*n > la {
						fmt.Printf(" *** In Ddrvgb, LA=%5d is too small for n=%5d, ku=%5d, kl=%5d\n ==> Increase LA to at least %5d\n", la, n, kl, ku, n*(kl+ku+1))
						nerrs = nerrs + 1
					}
					if ldafb*n > lafb {
						fmt.Printf(" *** In Ddrvgb, LAFB=%5d is too small for n=%5d, ku=%5d, kl=%5d\n ==> Increase LAFB to at least %5d\n", lafb, n, kl, ku, n*(2*kl+ku+1))
						nerrs++
					}
					goto label130
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !dotype[imat-1] {
						goto label120
					}

					//                 Skip types 2, 3, or 4 if the matrix is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label120
					}

					//                 Set up parameters with DLATB4 and generate a
					//                 test matrix with DLATMS.
					_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)
					rcondc = one / cndnum
					//
					*srnamt = "Dlatms"
					if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'Z', a.Matrix(lda, opts), work); info != 0 {
						nerrs = alaerh(path, "Dlatms", info, 0, []byte(" "), n, n, kl, ku, -1, imat, nfail, nerrs)
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
							i1 = max(1, ku+2-izero)
							i2 = min(kl+ku+1, ku+1+(n-izero))
							for i = i1; i <= i2; i++ {
								a.Set(ioff+i-1, zero)
							}
						} else {
							for j = izero; j <= n; j++ {
								for i = max(1, ku+2-j); i <= min(kl+ku+1, ku+1+(n-j)); i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						}
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Dlacpy(Full, kl+ku+1, n, a.Matrix(lda, opts), asav.Matrix(lda, opts))

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
								golapack.Dlacpy(Full, kl+ku+1, n, asav.Matrix(lda, opts), afb.MatrixOff(kl, ldafb, opts))
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									if rowcnd, colcnd, amax, info, err = golapack.Dgbequ(n, n, kl, ku, afb.MatrixOff(kl, ldafb, opts), s, s.Off(n)); err != nil {
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

										//                                Equilibrate the matrix.
										equed = golapack.Dlaqgb(n, n, kl, ku, afb.MatrixOff(kl, ldafb, opts), s, s.Off(n), rowcnd, colcnd, amax)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in DGET04.
								if equil {
									roldo = rcondo
									roldi = rcondi
								}

								//                          Compute the 1-norm and infinity-norm of A.
								anormo = golapack.Dlangb('1', n, kl, ku, afb.MatrixOff(kl, ldafb, opts), rwork)
								anormi = golapack.Dlangb('I', n, kl, ku, afb.MatrixOff(kl, ldafb, opts), rwork)

								//                          Factor the matrix A.
								if info, err = golapack.Dgbtrf(n, n, kl, ku, afb.Matrix(ldafb, opts), &iwork); err != nil {
									panic(err)
								}

								//                          Form the inverse of A.
								golapack.Dlaset(Full, n, n, zero, one, work.Matrix(ldb, opts))
								*srnamt = "Dgbtrs"
								if err = golapack.Dgbtrs(NoTrans, n, kl, ku, n, afb.Matrix(ldafb, opts), iwork, work.Matrix(ldb, opts)); err != nil {
									panic(err)
								}

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Dlange('1', n, n, work.Matrix(ldb, opts), rwork)
								if anormo <= zero || ainvnm <= zero {
									rcondo = one
								} else {
									rcondo = (one / anormo) / ainvnm
								}

								//                          Compute the infinity-norm condition number
								//                          of A.
								ainvnm = golapack.Dlange('I', n, n, work.Matrix(ldb, opts), rwork)
								if anormi <= zero || ainvnm <= zero {
									rcondi = one
								} else {
									rcondi = (one / anormi) / ainvnm
								}
							}

							for _, trans = range mat.IterMatTrans() {
								//                          Do for each value of TRANS.
								if trans == NoTrans {
									rcondc = rcondo
								} else {
									rcondc = rcondi
								}

								//                          Restore the matrix A.
								golapack.Dlacpy(Full, kl+ku+1, n, asav.Matrix(lda, opts), a.Matrix(lda, opts))

								//                          Form an exact solution and set the right hand
								//                          side.
								*srnamt = "Dlarhs"
								if err = Dlarhs(path, xtype, Full, trans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(ldb, opts), b.Matrix(ldb, opts), &iseed); err != nil {
									panic(err)
								}
								xtype = 'C'
								golapack.Dlacpy(Full, n, nrhs, b.Matrix(ldb, opts), bsav.Matrix(ldb, opts))

								if nofact && trans == NoTrans {
									//                             --- Test Dgbsv ---
									//
									//                             Compute the LU factorization of the matrix
									//                             and solve the system.
									golapack.Dlacpy(Full, kl+ku+1, n, a.Matrix(lda, opts), afb.MatrixOff(kl, ldafb, opts))
									golapack.Dlacpy(Full, n, nrhs, b.Matrix(ldb, opts), x.Matrix(ldb, opts))

									*srnamt = "Dgbsv"
									info, err = golapack.Dgbsv(n, kl, ku, nrhs, afb.Matrix(ldafb, opts), &iwork, x.Matrix(ldb, opts))

									//                             Check error code from Dgbsv.
									if info != izero {
										nerrs = alaerh(path, "Dgbsv", info, 0, []byte(" "), n, n, kl, ku, nrhs, imat, nfail, nerrs)
									}

									//                             Reconstruct matrix from factors and
									//                             compute residual.
									result.Set(0, dgbt01(n, n, kl, ku, a.Matrix(lda, opts), afb.Matrix(ldafb, opts), iwork, work))
									nt = 1
									if izero == 0 {
										//                                Compute residual of the computed
										//                                solution.
										golapack.Dlacpy(Full, n, nrhs, b.Matrix(ldb, opts), work.Matrix(ldb, opts))
										result.Set(1, dgbt02(NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), x.Matrix(ldb, opts), work.Matrix(ldb, opts)))

										//                                Check solution from generated exact
										//                                solution.
										result.Set(2, dget04(n, nrhs, x.Matrix(ldb, opts), xact.Matrix(ldb, opts), rcondc))
										nt = 3
									}

									//                             Print information about the tests that did
									//                             not pass the threshold.
									for k = 1; k <= nt; k++ {
										if result.Get(k-1) >= thresh {
											if nfail == 0 && nerrs == 0 {
												aladhd(path)
											}
											fmt.Printf(" %s, n=%5d, kl=%5d, ku=%5d, _type %1d, test(%1d)=%12.5f\n", "Dgbsv", n, kl, ku, imat, k, result.Get(k-1))
											nfail++
										}
									}
									nrun += nt
								}

								//                          --- Test Dgbsvx ---
								if !prefac {
									golapack.Dlaset(Full, 2*kl+ku+1, n, zero, zero, afb.Matrix(ldafb, opts))
								}
								golapack.Dlaset(Full, n, nrhs, zero, zero, x.Matrix(ldb, opts))
								if iequed > 1 && n > 0 {
									//                             Equilibrate the matrix if FACT = 'F' and
									//                             equed = 'R', 'C', or 'B'.
									equed = golapack.Dlaqgb(n, n, kl, ku, a.Matrix(lda, opts), s, s.Off(n), rowcnd, colcnd, amax)
								}

								//                          Solve the system and compute the condition
								//                          number and error bounds using Dgbsvx.
								*srnamt = "Dgbsvx"
								equed, rcond, info, err = golapack.Dgbsvx(fact, trans, n, kl, ku, nrhs, a.Matrix(lda, opts), afb.Matrix(ldafb, opts), &iwork, equed, s, s.Off(n), b.Matrix(ldb, opts), x.Matrix(ldb, opts), rwork, rwork.Off(nrhs), work, toSlice(&iwork, n))

								//                          Check the error code from Dgbsvx.
								if info != izero {
									nerrs = alaerh(path, "Dgbsvx", info, 0, []byte{fact, trans.Byte()}, n, n, kl, ku, nrhs, imat, nfail, nerrs)
								}

								//                          Compare WORK(1) from Dgbsvx with the computed
								//                          reciprocal pivot growth factor RPVGRW
								if info != 0 && info <= n {
									anrmpv = zero
									for j = 1; j <= info; j++ {
										for i = max(ku+2-j, 1); i <= min(n+ku+1-j, kl+ku+1); i++ {
											anrmpv = math.Max(anrmpv, math.Abs(a.Get(i-1+(j-1)*lda)))
										}
									}
									rpvgrw = golapack.Dlantb('M', Upper, NonUnit, info, min(info-1, kl+ku), afb.MatrixOff(max(1, kl+ku+2-info)-1, ldafb, opts), work)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = anrmpv / rpvgrw
									}
								} else {
									rpvgrw = golapack.Dlantb('M', Upper, NonUnit, n, kl+ku, afb.Matrix(ldafb, opts), work)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = golapack.Dlangb('M', n, kl, ku, a.Matrix(lda, opts), work) / rpvgrw
									}
								}
								result.Set(6, math.Abs(rpvgrw-work.Get(0))/math.Max(work.Get(0), rpvgrw)/golapack.Dlamch(Epsilon))

								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									result.Set(0, dgbt01(n, n, kl, ku, a.Matrix(lda, opts), afb.Matrix(ldafb, opts), iwork, work))
									k1 = 1
								} else {
									k1 = 2
								}

								if info == 0 {
									trfcon = false

									//                             Compute residual of the computed solution.
									golapack.Dlacpy(Full, n, nrhs, bsav.Matrix(ldb, opts), work.Matrix(ldb, opts))
									result.Set(1, dgbt02(trans, n, n, kl, ku, nrhs, asav.Matrix(lda, opts), x.Matrix(ldb, opts), work.Matrix(ldb, opts)))

									//                             Check solution from generated exact
									//                             solution.
									if nofact || (prefac && equed == 'N') {
										result.Set(2, dget04(n, nrhs, x.Matrix(ldb, opts), xact.Matrix(ldb, opts), rcondc))
									} else {
										if trans == NoTrans {
											roldc = roldo
										} else {
											roldc = roldi
										}
										result.Set(2, dget04(n, nrhs, x.Matrix(ldb, opts), xact.Matrix(ldb, opts), roldc))
									}

									//                             Check the error bounds from iterative
									//                             refinement.
									dgbt05(trans, n, kl, ku, nrhs, asav.Matrix(lda, opts), b.Matrix(ldb, opts), x.Matrix(ldb, opts), xact.Matrix(ldb, opts), rwork, rwork.Off(nrhs), result.Off(3))
								} else {
									trfcon = true
								}

								//                          Compare RCOND from Dgbsvx with the computed
								//                          value in RCONDC.
								result.Set(5, dget06(rcond, rcondc))

								//                          Print information about the tests that did
								//                          not pass the threshold.
								if !trfcon {
									for k = k1; k <= ntests; k++ {
										if result.Get(k-1) >= thresh {
											if nfail == 0 && nerrs == 0 {
												aladhd(path)
											}
											t.Fail()
											if prefac {
												fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, equed, imat, k, result.Get(k-1))
											} else {
												fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, imat, k, result.Get(k-1))
											}
											nfail++
										}
									}
									nrun += ntests - k1 + 1
								} else {
									if result.Get(0) >= thresh && !prefac {
										if nfail == 0 && nerrs == 0 {
											aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, equed, imat, 1, result.Get(0))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, imat, 1, result.Get(0))
										}
										nfail++
										nrun++
									}
									if result.Get(5) >= thresh {
										if nfail == 0 && nerrs == 0 {
											aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, equed, imat, 6, result.Get(5))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, imat, 6, result.Get(5))
										}
										nfail++
										nrun++
									}
									if result.Get(6) >= thresh {
										if nfail == 0 && nerrs == 0 {
											aladhd(path)
										}
										t.Fail()
										if prefac {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, equed, imat, 7, result.Get(6))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Dgbsvx", fact, trans, n, kl, ku, imat, 7, result.Get(6))
										}
										nfail++
										nrun++
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
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
