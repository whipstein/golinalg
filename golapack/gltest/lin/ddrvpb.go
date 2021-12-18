package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// ddrvpb tests the driver routines DPBSVand -SVX.
func ddrvpb(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, afac, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var equil, nofact, prefac, zerot bool
	var dist, equed, fact, packit, _type, xtype byte
	var uplo mat.MatUplo
	var ainvnm, amax, anorm, cndnum, one, rcond, rcondc, roldc, scond, zero float64
	var i, i1, i2, iequed, ifact, ikd, imat, in, info, ioff, iw, izero, k, k1, kd, koff, lda, ldab, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nkd, nrun, nt, ntypes int
	var err error

	equeds := make([]byte, 2)
	facts := make([]byte, 3)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kdval := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 8

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1] = 'N', 'Y'

	//     Initialize constants and the random number seed.
	path := "Dpb"
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
	kdval[0] = 0

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

		//        Set limits on the number of loop iterations.
		//
		nkd = max(1, min(n, 4))
		nimat = ntypes
		if n == 0 {
			nimat = 1
		}

		kdval[1] = n + (n+1)/4
		kdval[2] = (3*n - 1) / 4
		kdval[3] = (n + 1) / 4

		for ikd = 1; ikd <= nkd; ikd++ {
			//           Do for KD = 0, (5*N+1)/4, (3N-1)/4, and (N+1)/4. This order
			//           makes it easier to skip redundant values for small values
			//           of N.
			kd = kdval[ikd-1]
			ldab = kd + 1

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for _, uplo = range mat.IterMatUplo(false) {
				koff = 1
				if uplo == Upper {
					packit = 'Q'
					koff = max(1, kd+2-n)
				} else {
					packit = 'B'
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !dotype[imat-1] {
						goto label80
					}

					//                 Skip types 2, 3, or 4 if the matrix size is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label80
					}

					if !zerot || !dotype[0] {
						//                    Set up parameters with DLATB4 and generate a test
						//                    matrix with DLATMS.
						_type, _, _, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)

						*srnamt = "Dlatms"
						if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kd, kd, packit, a.Off(koff-1).Matrix(ldab, opts), work); info != 0 {
							nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
							goto label80
						}
					} else if izero > 0 {
						//                    Use the same matrix for types 3 and 4 as for _type
						//                    2 by copying back the zeroed out column,
						iw = 2*lda + 1
						if uplo == Upper {
							ioff = (izero-1)*ldab + kd + 1
							a.Off(ioff-izero+i1-1).Copy(izero-i1, work.Off(iw-1), 1, 1)
							iw = iw + izero - i1
							a.Off(ioff-1).Copy(i2-izero+1, work.Off(iw-1), 1, max(ldab-1, 1))
						} else {
							ioff = (i1-1)*ldab + 1
							a.Off(ioff+izero-i1-1).Copy(izero-i1, work.Off(iw-1), 1, max(ldab-1, 1))
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							a.Off(ioff-1).Copy(i2-izero+1, work.Off(iw-1), 1, 1)
						}
					}

					//                 For types 2-4, zero one row and column of the matrix
					//                 to test that INFO is returned correctly.
					izero = 0
					if zerot {
						if imat == 2 {
							izero = 1
						} else if imat == 3 {
							izero = n
						} else {
							izero = n/2 + 1
						}

						//                    Save the zeroed out row and column in WORK(*,3)
						iw = 2 * lda
						for i = 1; i <= min(2*kd+1, n); i++ {
							work.Set(iw+i-1, zero)
						}
						iw = iw + 1
						i1 = max(izero-kd, 1)
						i2 = min(izero+kd, n)
						//
						if uplo == Upper {
							ioff = (izero-1)*ldab + kd + 1
							work.Off(iw-1).Swap(izero-i1, a.Off(ioff-izero+i1-1), 1, 1)
							iw = iw + izero - i1
							work.Off(iw-1).Swap(i2-izero+1, a.Off(ioff-1), max(ldab-1, 1), 1)
						} else {
							ioff = (i1-1)*ldab + 1
							work.Off(iw-1).Swap(izero-i1, a.Off(ioff+izero-i1-1), max(ldab-1, 1), 1)
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							work.Off(iw-1).Swap(i2-izero+1, a.Off(ioff-1), 1, 1)
						}
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Dlacpy(Full, kd+1, n, a.Matrix(ldab, opts), asav.Matrix(ldab, opts))

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
									goto label60
								}
								rcondc = zero

							} else if fact != 'N' {
								//                          Compute the condition number for comparison
								//                          with the value returned by Dpbsvx (FACT =
								//                          'N' reuses the condition number from the
								//                          previous iteration with FACT = 'F').
								golapack.Dlacpy(Full, kd+1, n, asav.Matrix(ldab, opts), afac.Matrix(ldab, opts))
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									if scond, amax, info, err = golapack.Dpbequ(uplo, n, kd, afac.Matrix(ldab, opts), s); err != nil {
										panic(err)
									}
									if info == 0 && n > 0 {
										if iequed > 1 {
											scond = zero
										}

										//                                Equilibrate the matrix.
										equed = golapack.Dlaqsb(uplo, n, kd, afac.Matrix(ldab, opts), s, scond, amax)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in DGET04.
								if equil {
									roldc = rcondc
								}

								//                          Compute the 1-norm of A.
								anorm = golapack.Dlansb('1', uplo, n, kd, afac.Matrix(ldab, opts), rwork)

								//                          Factor the matrix A.
								if info, err = golapack.Dpbtrf(uplo, n, kd, afac.Matrix(ldab, opts)); err != nil {
									panic(err)
								}

								//                          Form the inverse of A.
								golapack.Dlaset(Full, n, n, zero, one, a.Matrix(lda, opts))
								*srnamt = "Dpbtrs"
								if err = golapack.Dpbtrs(uplo, n, kd, n, afac.Matrix(ldab, opts), a.Matrix(lda, opts)); err != nil {
									panic(err)
								}

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Dlange('1', n, n, a.Matrix(lda, opts), rwork)
								if anorm <= zero || ainvnm <= zero {
									rcondc = one
								} else {
									rcondc = (one / anorm) / ainvnm
								}
							}

							//                       Restore the matrix A.
							golapack.Dlacpy(Full, kd+1, n, asav.Matrix(ldab, opts), a.Matrix(ldab, opts))

							//                       Form an exact solution and set the right hand
							//                       side.
							*srnamt = "Dlarhs"
							if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kd, kd, nrhs, a.Matrix(ldab, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), bsav.Matrix(lda, opts))

							if nofact {
								//                          --- Test Dpbsv ---
								//
								//                          Compute the L*L' or U'*U factorization of the
								//                          matrix and solve the system.
								golapack.Dlacpy(Full, kd+1, n, a.Matrix(ldab, opts), afac.Matrix(ldab, opts))
								golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

								*srnamt = "Dpbsv"
								if info, err = golapack.Dpbsv(uplo, n, kd, nrhs, afac.Matrix(ldab, opts), x.Matrix(lda, opts)); err != nil {
									panic(err)
								}

								//                          Check error code from Dpbsv.
								if info != izero {
									nerrs = alaerh(path, "Dpbsv", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
									goto label40
								} else if info != 0 {
									goto label40
								}

								//                          Reconstruct matrix from factors and compute
								//                          residual.
								result.Set(0, dpbt01(uplo, n, kd, a.Matrix(ldab, opts), afac.Matrix(ldab, opts), rwork))

								//                          Compute residual of the computed solution.
								golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
								result.Set(1, dpbt02(uplo, n, kd, nrhs, a.Matrix(ldab, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

								//                          Check solution from generated exact solution.
								result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
								nt = 3

								//                          Print information about the tests that did
								//                          not pass the threshold.
								for k = 1; k <= nt; k++ {
									if result.Get(k-1) >= thresh {
										if nfail == 0 && nerrs == 0 {
											aladhd(path)
										}
										t.Fail()
										fmt.Printf(" %s, UPLO='%c', N =%5d, KD =%5d, _type %1d, test(%1d)=%12.5f\n", "Dpbsv", uplo, n, kd, imat, k, result.Get(k-1))
										nfail++
									}
								}
								nrun = nrun + nt
							label40:
							}

							//                       --- Test Dpbsvx ---
							if !prefac {
								golapack.Dlaset(Full, kd+1, n, zero, zero, afac.Matrix(ldab, opts))
							}
							golapack.Dlaset(Full, n, nrhs, zero, zero, x.Matrix(lda, opts))
							if iequed > 1 && n > 0 {
								//                          Equilibrate the matrix if FACT='F' and
								//                          EQUED='Y'
								equed = golapack.Dlaqsb(uplo, n, kd, a.Matrix(ldab, opts), s, scond, amax)
							}

							//                       Solve the system and compute the condition
							//                       number and error bounds using Dpbsvx.
							*srnamt = "Dpbsvx"
							if equed, rcond, info, err = golapack.Dpbsvx(fact, uplo, n, kd, nrhs, a.Matrix(ldab, opts), afac.Matrix(ldab, opts), equed, s, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
								panic(err)
							}

							//                       Check the error code from Dpbsvx.
							if info != izero {
								nerrs = alaerh(path, "Dpbsvx", info, 0, []byte{fact, uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
								goto label60
							}

							if info == 0 {
								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									result.Set(0, dpbt01(uplo, n, kd, a.Matrix(ldab, opts), afac.Matrix(ldab, opts), rwork.Off(2*nrhs)))
									k1 = 1
								} else {
									k1 = 2
								}

								//                          Compute residual of the computed solution.
								golapack.Dlacpy(Full, n, nrhs, bsav.Matrix(lda, opts), work.Matrix(lda, opts))
								result.Set(1, dpbt02(uplo, n, kd, nrhs, asav.Matrix(ldab, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork.Off(2*nrhs)))

								//                          Check solution from generated exact solution.
								if nofact || (prefac && equed == 'N') {
									result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
								} else {
									result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), roldc))
								}

								//                          Check the error bounds from iterative
								//                          refinement.
								dpbt05(uplo, n, kd, nrhs, asav.Matrix(ldab, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
							} else {
								k1 = 6
							}

							//                       Compare RCOND from Dpbsvx with the computed
							//                       value in RCONDC.
							result.Set(5, dget06(rcond, rcondc))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = k1; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										aladhd(path)
									}
									t.Fail()
									if prefac {
										fmt.Printf(" %s( '%c', %s, %5d, %5d, ... ), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "Dpbsvx", fact, uplo, n, kd, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s( '%c', %s, %5d, %5d, ... ), _type %1d, test(%1d)=%12.5f\n", "Dpbsvx", fact, uplo, n, kd, imat, k, result.Get(k-1))
									}
									nfail++
								}
							}
							nrun += 7 - k1
						label60:
						}
					}
				label80:
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 4750
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
