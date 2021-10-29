package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// ddrvpo tests the driver routines Dposvand -SVX.
func ddrvpo(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, afac, asav, b, bsav, x, xact, s, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var equil, nofact, prefac, zerot bool
	var dist, equed, fact, _type, xtype byte
	var uplo mat.MatUplo
	var ainvnm, amax, anorm, cndnum, one, rcond, rcondc, roldc, scond, zero float64
	var i, iequed, ifact, imat, in, info, ioff, izero, k, k1, kl, ku, lda, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntypes int
	var err error

	equeds := make([]byte, 2)
	facts := make([]byte, 3)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 9

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2] = 'F', 'N', 'E'
	equeds[0], equeds[1] = 'N', 'Y'

	//     Initialize constants and the random number seed.
	path := "Dpo"
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
	xlaenv(1, nbmin)

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
				goto label120
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label120
			}

			//           Do first for uplo = 'U', then for uplo = 'L'
			for _, uplo = range mat.IterMatUplo(false) {

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)

				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
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
					if uplo == Upper {
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
				golapack.Dlacpy(uplo, n, n, a.Matrix(lda, opts), asav.Matrix(lda, opts))

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
							golapack.Dlacpy(uplo, n, n, asav.Matrix(lda, opts), afac.Matrix(lda, opts))
							if equil || iequed > 1 {
								//                          Compute row and column scale factors to
								//                          equilibrate the matrix A.
								if scond, amax, info, err = golapack.Dpoequ(n, afac.Matrix(lda, opts), s); err != nil {
									panic(err)
								}
								if info == 0 && n > 0 {
									if iequed > 1 {
										scond = zero
									}

									//                             Equilibrate the matrix.
									equed = golapack.Dlaqsy(uplo, n, afac.Matrix(lda, opts), s, scond, amax)
								}
							}

							//                       Save the condition number of the
							//                       non-equilibrated system for use in DGET04.
							if equil {
								roldc = rcondc
							}

							//                       Compute the 1-norm of A.
							anorm = golapack.Dlansy('1', uplo, n, afac.Matrix(lda, opts), rwork)

							//                       Factor the matrix A.
							if info, err = golapack.Dpotrf(uplo, n, afac.Matrix(lda, opts)); err != nil {
								panic(err)
							}

							//                       Form the inverse of A.
							golapack.Dlacpy(uplo, n, n, afac.Matrix(lda, opts), a.Matrix(lda, opts))
							if info, err = golapack.Dpotri(uplo, n, a.Matrix(lda, opts)); err != nil {
								panic(err)
							}

							//                       Compute the 1-norm condition number of A.
							ainvnm = golapack.Dlansy('1', uplo, n, a.Matrix(lda, opts), rwork)
							if anorm <= zero || ainvnm <= zero {
								rcondc = one
							} else {
								rcondc = (one / anorm) / ainvnm
							}
						}

						//                    Restore the matrix A.
						golapack.Dlacpy(uplo, n, n, asav.Matrix(lda, opts), a.Matrix(lda, opts))

						//                    Form an exact solution and set the right hand side.
						*srnamt = "Dlarhs"
						if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						xtype = 'C'
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), bsav.Matrix(lda, opts))

						if nofact {
							//                       --- Test Dposv ---
							//
							//                       Compute the L*L' or U'*U factorization of the
							//                       matrix and solve the system.
							golapack.Dlacpy(uplo, n, n, a.Matrix(lda, opts), afac.Matrix(lda, opts))
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

							*srnamt = "Dposv"
							if info, err = golapack.Dposv(uplo, n, nrhs, afac.Matrix(lda, opts), x.Matrix(lda, opts)); err != nil {
								panic(err)
							}

							//                       Check error code from Dposv.
							if info != izero {
								nerrs = alaerh(path, "Dposv", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
								goto label70
							} else if info != 0 {
								goto label70
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							result.Set(0, dpot01(uplo, n, a.Matrix(lda, opts), afac.Matrix(lda, opts), rwork))

							//                       Compute residual of the computed solution.
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
							result.Set(1, dpot02(uplo, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

							//                       Check solution from generated exact solution.
							result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							nt = 3

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										aladhd(path)
									}
									t.Fail()
									fmt.Printf(" %s, uplo='%c', n=%5d, _type %1d, test(%1d)=%12.5f\n", "Dposv", uplo, n, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun = nrun + nt
						label70:
						}

						//                    --- Test DPOSVX ---
						if !prefac {
							golapack.Dlaset(uplo, n, n, zero, zero, afac.Matrix(lda, opts))
						}
						golapack.Dlaset(Full, n, nrhs, zero, zero, x.Matrix(lda, opts))
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT='F' and
							//                       EQUED='Y'.
							equed = golapack.Dlaqsy(uplo, n, a.Matrix(lda, opts), s, scond, amax)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using DPOSVX.
						*srnamt = "Dposvx"
						if equed, rcond, info, err = golapack.Dposvx(fact, uplo, n, nrhs, a.Matrix(lda, opts), afac.Matrix(lda, opts), equed, s, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
							panic(err)
						}

						//                    Check the error code from DPOSVX.
						if info != izero {
							nerrs = alaerh(path, "Dposvx", info, 0, []byte{fact, uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							goto label90
						}

						if info == 0 {
							if !prefac {
								//                          Reconstruct matrix from factors and compute
								//                          residual.
								result.Set(0, dpot01(uplo, n, a.Matrix(lda, opts), afac.Matrix(lda, opts), rwork.Off(2*nrhs)))
								k1 = 1
							} else {
								k1 = 2
							}

							//                       Compute residual of the computed solution.
							golapack.Dlacpy(Full, n, nrhs, bsav.Matrix(lda, opts), work.Matrix(lda, opts))
							result.Set(1, dpot02(uplo, n, nrhs, asav.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork.Off(2*nrhs)))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							} else {
								result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), roldc))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							dpot05(uplo, n, nrhs, asav.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
						} else {
							k1 = 6
						}

						//                    Compare RCOND from DPOSVX with the computed value
						//                    in RCONDC.
						result.Set(5, dget06(rcond, rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = k1; k <= 6; k++ {
							if result.Get(k-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								t.Fail()
								if prefac {
									fmt.Printf(" %s, FACT='%c', uplo=%s, N=%5d, EQUED='%c', _type %1d, test(%1d) =%12.5f\n", "DPOSVX", fact, uplo, n, equed, imat, k, result.Get(k-1))
								} else {
									fmt.Printf(" %s, FACT='%c', uplo=%s, N=%5d, _type %1d, test(%1d)=%12.5f\n", "DPOSVX", fact, uplo, n, imat, k, result.Get(k-1))
								}
								nfail++
							}
						}
						nrun += 7 - k1
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
	alasvm(path, nfail, nrun, nerrs)
}
