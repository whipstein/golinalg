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

// zdrvgb tests the driver routines Zgbsvand -SVX.
func zdrvgb(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, a *mat.CVector, la int, afb *mat.CVector, lafb int, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var equil, nofact, prefac, trfcon, zerot bool
	var dist, equed, fact, _type, xtype byte
	var trans mat.MatTrans
	var ainvnm, amax, anorm, anormi, anormo, anrmpv, cndnum, colcnd, one, rcond, rcondc, rcondi, rcondo, roldc, roldi, roldo, rowcnd, rpvgrw, zero float64
	var i, i1, i2, iequed, ifact, ikl, iku, imat, in, info, ioff, izero, j, k, k1, kl, ku, lda, ldafb, ldb, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nkl, nku, nrun, nt, ntests, ntypes int
	var err error

	equeds := make([]byte, 4)
	facts := make([]byte, 3)
	rdum := vf(1)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 8
	ntests = 7
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1], equeds[2], equeds[3] = 'N', 'R', 'C', 'B'

	//     Initialize constants and the random number seed.
	path := "Zgb"
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
					if lda*n > la {
						t.Fail()
						fmt.Printf(" *** In zdrvgb, la=%5d is too small for n=%5d, ku=%5d, kl=%5d\n ==> Increase la to at least %5d\n", la, n, kl, ku, n*(kl+ku+1))
						nerrs = nerrs + 1
					}
					if ldafb*n > lafb {
						t.Fail()
						fmt.Printf(" *** In zdrvgb, lafb=%5d is too small for n=%5d, ku=%5d, kl=%5d\n ==> Increase lafb to at least %5d\n", lafb, n, kl, ku, n*(2*kl+ku+1))
						nerrs = nerrs + 1
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

					//                 Set up parameters with ZLATB4 and generate a
					//                 test matrix with Zlatms.
					_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)
					rcondc = one / cndnum

					*srnamt = "Zlatms"
					if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'Z', a.CMatrix(lda, opts), work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, n, n, kl, ku, -1, imat, nfail, nerrs)
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
								a.SetRe(ioff+i-1, zero)
							}
						} else {
							for j = izero; j <= n; j++ {
								for i = max(1, ku+2-j); i <= min(kl+ku+1, ku+1+(n-j)); i++ {
									a.SetRe(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						}
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Zlacpy(Full, kl+ku+1, n, a.CMatrix(lda, opts), asav.CMatrix(lda, opts))

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
								golapack.Zlacpy(Full, kl+ku+1, n, asav.CMatrix(lda, opts), afb.CMatrixOff(kl, ldafb, opts))
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									if rowcnd, colcnd, amax, info, err = golapack.Zgbequ(n, n, kl, ku, afb.CMatrixOff(kl, ldafb, opts), s, s.Off(n)); err != nil {
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
										equed = golapack.Zlaqgb(n, n, kl, ku, afb.CMatrixOff(kl, ldafb, opts), s, s.Off(n), rowcnd, colcnd, amax)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in ZGET04.
								if equil {
									roldo = rcondo
									roldi = rcondi
								}

								//                          Compute the 1-norm and infinity-norm of A.
								anormo = golapack.Zlangb('1', n, kl, ku, afb.CMatrixOff(kl, ldafb, opts), rwork)
								anormi = golapack.Zlangb('I', n, kl, ku, afb.CMatrixOff(kl, ldafb, opts), rwork)

								//                          Factor the matrix A.
								if info, err = golapack.Zgbtrf(n, n, kl, ku, afb.CMatrix(ldafb, opts), &iwork); err != nil {
									panic(err)
								}

								//                          Form the inverse of A.
								golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), work.CMatrix(ldb, opts))
								*srnamt = "Zgbtrs"
								if err = golapack.Zgbtrs(NoTrans, n, kl, ku, n, afb.CMatrix(ldafb, opts), &iwork, work.CMatrix(ldb, opts)); err != nil {
									panic(err)
								}

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Zlange('1', n, n, work.CMatrix(ldb, opts), rwork)
								if anormo <= zero || ainvnm <= zero {
									rcondo = one
								} else {
									rcondo = (one / anormo) / ainvnm
								}

								//                          Compute the infinity-norm condition number
								//                          of A.
								ainvnm = golapack.Zlange('I', n, n, work.CMatrix(ldb, opts), rwork)
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
								golapack.Zlacpy(Full, kl+ku+1, n, asav.CMatrix(lda, opts), a.CMatrix(lda, opts))

								//                          Form an exact solution and set the right hand
								//                          side.
								*srnamt = "zlarhs"
								if err = zlarhs(path, xtype, Full, trans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(ldb, opts), b.CMatrix(ldb, opts), &iseed); err != nil {
									panic(err)
								}
								xtype = 'C'
								golapack.Zlacpy(Full, n, nrhs, b.CMatrix(ldb, opts), bsav.CMatrix(ldb, opts))

								if nofact && trans == NoTrans {
									//                             --- Test Zgbsv ---
									//
									//                             Compute the LU factorization of the matrix
									//                             and solve the system.
									golapack.Zlacpy(Full, kl+ku+1, n, a.CMatrix(lda, opts), afb.CMatrixOff(kl, ldafb, opts))
									golapack.Zlacpy(Full, n, nrhs, b.CMatrix(ldb, opts), x.CMatrix(ldb, opts))

									*srnamt = "Zgbsv"
									if info, err = golapack.Zgbsv(n, kl, ku, nrhs, afb.CMatrix(ldafb, opts), &iwork, x.CMatrix(ldb, opts)); err != nil {
										panic(err)
									}

									//                             Check error code from Zgbsv.
									if info != izero {
										nerrs = alaerh(path, "Zgbsv", info, 0, []byte{' '}, n, n, kl, ku, nrhs, imat, nfail, nerrs)
									}

									//                             Reconstruct matrix from factors and
									//                             compute residual.
									*result.GetPtr(0) = zgbt01(n, n, kl, ku, a.CMatrix(lda, opts), afb.CMatrix(ldafb, opts), &iwork, work)
									nt = 1
									if izero == 0 {
										//                                Compute residual of the computed
										//                                solution.
										golapack.Zlacpy(Full, n, nrhs, b.CMatrix(ldb, opts), work.CMatrix(ldb, opts))
										*result.GetPtr(1) = zgbt02(NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), x.CMatrix(ldb, opts), work.CMatrix(ldb, opts))

										//                                Check solution from generated exact
										//                                solution.
										*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(ldb, opts), xact.CMatrix(ldb, opts), rcondc)
										nt = 3
									}

									//                             Print information about the tests that did
									//                             not pass the threshold.
									for k = 1; k <= nt; k++ {
										if result.Get(k-1) >= thresh {
											t.Fail()
											if nfail == 0 && nerrs == 0 {
												aladhd(path)
											}
											fmt.Printf(" %s, n=%5d, kl=%5d, ku=%5d, _type %1d, test(%1d)=%12.5f\n", "Zgbsv", n, kl, ku, imat, k, result.Get(k-1))
											nfail++
										}
									}
									nrun = nrun + nt
								}

								//                          --- Test Zgbsvx ---
								if !prefac {
									golapack.Zlaset(Full, 2*kl+ku+1, n, complex(zero, 0), complex(zero, 0), afb.CMatrix(ldafb, opts))
								}
								golapack.Zlaset(Full, n, nrhs, complex(zero, 0), complex(zero, 0), x.CMatrix(ldb, opts))
								if iequed > 1 && n > 0 {
									//                             Equilibrate the matrix if FACT = 'F' and
									//                             equed = 'R', 'C', or 'B'.
									equed = golapack.Zlaqgb(n, n, kl, ku, a.CMatrix(lda, opts), s, s.Off(n), rowcnd, colcnd, amax)
								}

								//                          Solve the system and compute the condition
								//                          number and error bounds using Zgbsvx.
								*srnamt = "Zgbsvx"
								if equed, rcond, info, err = golapack.Zgbsvx(fact, trans, n, kl, ku, nrhs, a.CMatrix(lda, opts), afb.CMatrix(ldafb, opts), &iwork, equed, s, s.Off(ldb), b.CMatrix(ldb, opts), x.CMatrix(ldb, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
									panic(err)
								}

								//                          Check the error code from Zgbsvx.
								if info != izero {
									nerrs = alaerh(path, "Zgbsvx", info, 0, append(append([]byte{}, fact), trans.Byte()), n, n, kl, ku, nrhs, imat, nfail, nerrs)
								}
								//                          Compare RWORK(2*nrhs+1) from Zgbsvx with the
								//                          computed reciprocal pivot growth RPVGRW
								if info != 0 && info <= n {
									anrmpv = zero
									for j = 1; j <= info; j++ {
										for i = max(ku+2-j, 1); i <= min(n+ku+1-j, kl+ku+1); i++ {
											anrmpv = math.Max(anrmpv, a.GetMag(i+(j-1)*lda-1))
										}
									}
									rpvgrw = golapack.Zlantb('M', Upper, NonUnit, info, min(info-1, kl+ku), afb.CMatrixOff(max(1, kl+ku+2-info)-1, ldafb, opts), rdum)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = anrmpv / rpvgrw
									}
								} else {
									rpvgrw = golapack.Zlantb('M', Upper, NonUnit, n, kl+ku, afb.CMatrix(ldafb, opts), rdum)
									if rpvgrw == zero {
										rpvgrw = one
									} else {
										rpvgrw = golapack.Zlangb('M', n, kl, ku, a.CMatrix(lda, opts), rdum) / rpvgrw
									}
								}
								result.Set(6, math.Abs(rpvgrw-rwork.Get(2*nrhs))/math.Max(rwork.Get(2*nrhs), rpvgrw)/golapack.Dlamch(Epsilon))

								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									*result.GetPtr(0) = zgbt01(n, n, kl, ku, a.CMatrix(lda, opts), afb.CMatrix(ldafb, opts), &iwork, work)
									k1 = 1
								} else {
									k1 = 2
								}

								if info == 0 {
									trfcon = false

									//                             Compute residual of the computed solution.
									golapack.Zlacpy(Full, n, nrhs, bsav.CMatrix(ldb, opts), work.CMatrix(ldb, opts))
									*result.GetPtr(1) = zgbt02(trans, n, n, kl, ku, nrhs, asav.CMatrix(lda, opts), x.CMatrix(ldb, opts), work.CMatrix(ldb, opts))

									//                             Check solution from generated exact
									//                             solution.
									if nofact || (prefac && equed == 'N') {
										*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(ldb, opts), xact.CMatrix(ldb, opts), rcondc)
									} else {
										if trans == NoTrans {
											roldc = roldo
										} else {
											roldc = roldi
										}
										*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(ldb, opts), xact.CMatrix(ldb, opts), roldc)
									}

									//                             Check the error bounds from iterative
									//                             refinement.
									zgbt05(trans, n, kl, ku, nrhs, asav.CMatrix(lda, opts), bsav.CMatrix(ldb, opts), x.CMatrix(ldb, opts), xact.CMatrix(ldb, opts), rwork, rwork.Off(nrhs), result.Off(3))
								} else {
									trfcon = true
								}

								//                          Compare RCOND from Zgbsvx with the computed
								//                          value in RCONDC.
								result.Set(5, dget06(rcond, rcondc))

								//                          Print information about the tests that did
								//                          not pass the threshold.
								if !trfcon {
									for k = k1; k <= ntests; k++ {
										if result.Get(k-1) >= thresh {
											t.Fail()
											if nfail == 0 && nerrs == 0 {
												aladhd(path)
											}
											if prefac {
												fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, equed, imat, k, result.Get(k-1))
											} else {
												fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, imat, k, result.Get(k-1))
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
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, equed, imat, 1, result.Get(0))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, imat, 1, result.Get(0))
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
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, equed, imat, 6, result.Get(5))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, imat, 6, result.Get(5))
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
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, equed, imat, 7, result.Get(6))
										} else {
											fmt.Printf(" %s( '%c',%s,%5d,%5d,%5d,...), _type %1d, test(%1d)=%12.5f\n", "Zgbsvx", fact, trans, n, kl, ku, imat, 7, result.Get(6))
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

	//     Print a summary of the results.
	alasvm(path, nfail, nrun, nerrs)
}
