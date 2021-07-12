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

// Zdrvpp tests the driver routines ZPPSV and -SVX.
func Zdrvpp(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
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

	one = 1.0
	zero = 0.0
	ntypes = 9
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], facts[0], facts[1], facts[2], packs[0], packs[1], equeds[0], equeds[1] = 'U', 'L', 'F', 'N', 'E', 'C', 'R', 'N', 'Y'

	//     Initialize constants and the random number seed.
	path := []byte("ZPP")
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

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = max(n, 1)
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

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with ZLATMS.
				Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
				rcondc = one / cndnum

				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
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
							a.SetRe(ioff+i-1, zero)
						}
						ioff = ioff + izero
						for i = izero; i <= n; i++ {
							a.SetRe(ioff-1, zero)
							ioff = ioff + i
						}
					} else {
						ioff = izero
						for i = 1; i <= izero-1; i++ {
							a.SetRe(ioff-1, zero)
							ioff = ioff + n - i
						}
						ioff = ioff - izero
						for i = izero; i <= n; i++ {
							a.SetRe(ioff+i-1, zero)
						}
					}
				} else {
					izero = 0
				}

				//              Set the imaginary part of the diagonals.
				if iuplo == 1 {
					Zlaipd(&n, a, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }())
				} else {
					Zlaipd(&n, a, &n, toPtr(-1))
				}

				//              Save a copy of the matrix A in ASAV.
				goblas.Zcopy(npp, a.Off(0, 1), asav.Off(0, 1))

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
								goto label100
							}
							rcondc = zero

						} else if fact != 'N' {

							//                       Compute the condition number for comparison with
							//                       the value returned by ZPPSVX (FACT = 'N' reuses
							//                       the condition number from the previous iteration
							//                          with FACT = 'F').
							goblas.Zcopy(npp, asav.Off(0, 1), afac.Off(0, 1))
							if equil || iequed > 1 {
								//                          Compute row and column scale factors to
								//                          equilibrate the matrix A.
								golapack.Zppequ(uplo, &n, afac, s, &scond, &amax, &info)
								if info == 0 && n > 0 {
									if iequed > 1 {
										scond = zero
									}

									//                             Equilibrate the matrix.
									golapack.Zlaqhp(uplo, &n, afac, s, &scond, &amax, &equed)
								}
							}

							//                       Save the condition number of the
							//                       non-equilibrated system for use in ZGET04.
							if equil {
								roldc = rcondc
							}

							//                       Compute the 1-norm of A.
							anorm = golapack.Zlanhp('1', uplo, &n, afac, rwork)

							//                       Factor the matrix A.
							golapack.Zpptrf(uplo, &n, afac, &info)

							//                       Form the inverse of A.
							goblas.Zcopy(npp, afac.Off(0, 1), a.Off(0, 1))
							golapack.Zpptri(uplo, &n, a, &info)

							//                       Compute the 1-norm condition number of A.
							ainvnm = golapack.Zlanhp('1', uplo, &n, a, rwork)
							if anorm <= zero || ainvnm <= zero {
								rcondc = one
							} else {
								rcondc = (one / anorm) / ainvnm
							}
						}

						//                    Restore the matrix A.
						goblas.Zcopy(npp, asav.Off(0, 1), a.Off(0, 1))

						//                    Form an exact solution and set the right hand side.
						*srnamt = "ZLARHS"
						Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, bsav.CMatrix(lda, opts), &lda)

						if nofact {
							//                       --- Test ZPPSV  ---
							//
							//                       Compute the L*L' or U'*U factorization of the
							//                       matrix and solve the system.
							goblas.Zcopy(npp, a.Off(0, 1), afac.Off(0, 1))
							golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

							*srnamt = "ZPPSV "
							golapack.Zppsv(uplo, &n, nrhs, afac, x.CMatrix(lda, opts), &lda, &info)

							//                       Check error code from ZPPSV .
							if info != izero {
								t.Fail()
								Alaerh(path, []byte("ZPPSV "), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
								goto label70
							} else if info != 0 {
								goto label70
							}

							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Zppt01(uplo, &n, a, afac, rwork, result.GetPtr(0))

							//                       Compute residual of the computed solution.
							golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
							Zppt02(uplo, &n, nrhs, a, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

							//                       Check solution from generated exact solution.
							Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							nt = 3

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= nt; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									fmt.Printf(" %s, UPLO='%c', N =%5d, _type %1d, test(%1d)=%12.5f\n", "ZPPSV ", uplo, n, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + nt
						label70:
						}

						//                    --- Test ZPPSVX ---
						if !prefac && npp > 0 {
							golapack.Zlaset('F', &npp, func() *int { y := 1; return &y }(), toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), afac.CMatrix(npp, opts), &npp)
						}
						golapack.Zlaset('F', &n, nrhs, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), x.CMatrix(lda, opts), &lda)
						if iequed > 1 && n > 0 {
							//                       Equilibrate the matrix if FACT='F' and
							//                       EQUED='Y'.
							golapack.Zlaqhp(uplo, &n, a, s, &scond, &amax, &equed)
						}

						//                    Solve the system and compute the condition number
						//                    and error bounds using ZPPSVX.
						*srnamt = "ZPPSVX"
						golapack.Zppsvx(fact, uplo, &n, nrhs, a, afac, &equed, s, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)), work, rwork.Off(2*(*nrhs)), &info)

						//                    Check the error code from ZPPSVX.
						if info != izero {
							t.Fail()
							Alaerh(path, []byte("ZPPSVX"), &info, &izero, []byte{fact, uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label90
						}

						if info == 0 {
							if !prefac {
								//                          Reconstruct matrix from factors and compute
								//                          residual.
								Zppt01(uplo, &n, a, afac, rwork.Off(2*(*nrhs)), result.GetPtr(0))
								k1 = 1
							} else {
								k1 = 2
							}

							//                       Compute residual of the computed solution.
							golapack.Zlacpy('F', &n, nrhs, bsav.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
							Zppt02(uplo, &n, nrhs, asav, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork.Off(2*(*nrhs)), result.GetPtr(1))

							//                       Check solution from generated exact solution.
							if nofact || (prefac && equed == 'N') {
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							} else {
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &roldc, result.GetPtr(2))
							}

							//                       Check the error bounds from iterative
							//                       refinement.
							Zppt05(uplo, &n, nrhs, asav, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)), result.Off(3))
						} else {
							k1 = 6
						}

						//                    Compare RCOND from ZPPSVX with the computed value
						//                    in RCONDC.
						result.Set(5, Dget06(&rcond, &rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = k1; k <= 6; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								if prefac {
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "ZPPSVX", fact, uplo, n, equed, imat, k, result.Get(k-1))
								} else {
									fmt.Printf(" %s, FACT='%c', UPLO='%c', N=%5d, _type %1d, test(%1d)=%12.5f\n", "ZPPSVX", fact, uplo, n, imat, k, result.Get(k-1))
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

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
