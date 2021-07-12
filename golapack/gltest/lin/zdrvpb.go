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

// Zdrvpb tests the driver routines ZPBSV and -SVX.
func Zdrvpb(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var equil, nofact, prefac, zerot bool
	var dist, equed, fact, packit, _type, uplo, xtype byte
	var ainvnm, amax, anorm, cndnum, one, rcond, rcondc, roldc, scond, zero float64
	var i, i1, i2, iequed, ifact, ikd, imat, in, info, ioff, iuplo, iw, izero, k, k1, kd, kl, koff, ku, lda, ldab, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nkd, nrun, nt, ntypes int

	equeds := make([]byte, 2)
	facts := make([]byte, 3)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kdval := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 8
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2], equeds[0], equeds[1] = 'F', 'N', 'E', 'N', 'Y'

	//     Initialize constants and the random number seed.
	path := []byte("ZPB")
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
	kdval[0] = 0

	//     Set the block size and minimum block size for testing.
	nb = 1
	nbmin = 2
	Xlaenv(1, nb)
	Xlaenv(2, nbmin)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = max(n, 1)
		xtype = 'N'

		//        Set limits on the number of loop iterations.
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
			for iuplo = 1; iuplo <= 2; iuplo++ {
				koff = 1
				if iuplo == 1 {
					uplo = 'U'
					packit = 'Q'
					koff = max(1, kd+2-n)
				} else {
					uplo = 'L'
					packit = 'B'
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !(*dotype)[imat-1] {
						goto label80
					}

					//                 Skip types 2, 3, or 4 if the matrix size is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label80
					}

					if !zerot || !(*dotype)[0] {
						//                    Set up parameters with ZLATB4 and generate a test
						//                    matrix with ZLATMS.
						Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

						*srnamt = "ZLATMS"
						matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kd, &kd, packit, a.CMatrixOff(koff-1, ldab, opts), &ldab, work, &info)

						//                    Check error code from ZLATMS.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
							goto label80
						}
					} else if izero > 0 {
						//                    Use the same matrix for types 3 and 4 as for _type
						//                    2 by copying back the zeroed out column,
						iw = 2*lda + 1
						if iuplo == 1 {
							ioff = (izero-1)*ldab + kd + 1
							goblas.Zcopy(izero-i1, work.Off(iw-1, 1), a.Off(ioff-izero+i1-1, 1))
							iw = iw + izero - i1
							goblas.Zcopy(i2-izero+1, work.Off(iw-1, 1), a.Off(ioff-1, max(ldab-1, 1)))
						} else {
							ioff = (i1-1)*ldab + 1
							goblas.Zcopy(izero-i1, work.Off(iw-1, 1), a.Off(ioff+izero-i1-1, max(ldab-1, 1)))
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							goblas.Zcopy(i2-izero+1, work.Off(iw-1, 1), a.Off(ioff-1, 1))
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
							work.SetRe(iw+i-1, zero)
						}
						iw = iw + 1
						i1 = max(izero-kd, 1)
						i2 = min(izero+kd, n)

						if iuplo == 1 {
							ioff = (izero-1)*ldab + kd + 1
							goblas.Zswap(izero-i1, a.Off(ioff-izero+i1-1, 1), work.Off(iw-1, 1))
							iw = iw + izero - i1
							goblas.Zswap(i2-izero+1, a.Off(ioff-1, max(ldab-1, 1)), work.Off(iw-1, 1))
						} else {
							ioff = (i1-1)*ldab + 1
							goblas.Zswap(izero-i1, a.Off(ioff+izero-i1-1, max(ldab-1, 1)), work.Off(iw-1, 1))
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							goblas.Zswap(i2-izero+1, a.Off(ioff-1, 1), work.Off(iw-1, 1))
						}
					}

					//                 Set the imaginary part of the diagonals.
					if iuplo == 1 {
						Zlaipd(&n, a.Off(kd), &ldab, func() *int { y := 0; return &y }())
					} else {
						Zlaipd(&n, a.Off(0), &ldab, func() *int { y := 0; return &y }())
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Zlacpy('F', toPtr(kd+1), &n, a.CMatrix(ldab, opts), &ldab, asav.CMatrix(ldab, opts), &ldab)

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
								//                          with the value returned by ZPBSVX (FACT =
								//                          'N' reuses the condition number from the
								//                          previous iteration with FACT = 'F').
								golapack.Zlacpy('F', toPtr(kd+1), &n, asav.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab)
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									golapack.Zpbequ(uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, s, &scond, &amax, &info)
									if info == 0 && n > 0 {
										if iequed > 1 {
											scond = zero
										}

										//                                Equilibrate the matrix.
										golapack.Zlaqhb(uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, s, &scond, &amax, &equed)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in ZGET04.
								if equil {
									roldc = rcondc
								}

								//                          Compute the 1-norm of A.
								anorm = golapack.Zlanhb('1', uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, rwork)

								//                          Factor the matrix A.
								golapack.Zpbtrf(uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, &info)

								//                          Form the inverse of A.
								golapack.Zlaset('F', &n, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), a.CMatrix(lda, opts), &lda)
								*srnamt = "ZPBTRS"
								golapack.Zpbtrs(uplo, &n, &kd, &n, afac.CMatrix(ldab, opts), &ldab, a.CMatrix(lda, opts), &lda, &info)

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Zlange('1', &n, &n, a.CMatrix(lda, opts), &lda, rwork)
								if anorm <= zero || ainvnm <= zero {
									rcondc = one
								} else {
									rcondc = (one / anorm) / ainvnm
								}
							}

							//                       Restore the matrix A.
							golapack.Zlacpy('F', toPtr(kd+1), &n, asav.CMatrix(ldab, opts), &ldab, a.CMatrix(ldab, opts), &ldab)

							//                       Form an exact solution and set the right hand
							//                       side.
							*srnamt = "ZLARHS"
							Zlarhs(path, xtype, uplo, ' ', &n, &n, &kd, &kd, nrhs, a.CMatrix(ldab, opts), &ldab, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
							xtype = 'C'
							golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, bsav.CMatrix(lda, opts), &lda)

							if nofact {
								//                          --- Test ZPBSV  ---
								//
								//                          Compute the L*L' or U'*U factorization of the
								//                          matrix and solve the system.
								golapack.Zlacpy('F', toPtr(kd+1), &n, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab)
								golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

								*srnamt = "ZPBSV "
								golapack.Zpbsv(uplo, &n, &kd, nrhs, afac.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, &info)

								//                          Check error code from ZPBSV .
								if info != izero {
									t.Fail()
									Alaerh(path, []byte("ZPBSV "), &info, &izero, []byte{uplo}, &n, &n, &kd, &kd, nrhs, &imat, &nfail, &nerrs)
									goto label40
								} else if info != 0 {
									goto label40
								}

								//                          Reconstruct matrix from factors and compute
								//                          residual.
								Zpbt01(uplo, &n, &kd, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab, rwork, result.GetPtr(0))

								//                          Compute residual of the computed solution.
								golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
								Zpbt02(uplo, &n, &kd, nrhs, a.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

								//                          Check solution from generated exact solution.
								Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
								nt = 3

								//                          Print information about the tests that did
								//                          not pass the threshold.
								for k = 1; k <= nt; k++ {
									if result.Get(k-1) >= (*thresh) {
										t.Fail()
										if nfail == 0 && nerrs == 0 {
											Aladhd(path)
										}
										fmt.Printf(" %s, UPLO='%c', N =%5d, KD =%5d, _type %1d, test(%1d)=%12.5f\n", "ZPBSV ", uplo, n, kd, imat, k, result.Get(k-1))
										nfail = nfail + 1
									}
								}
								nrun = nrun + nt
							label40:
							}

							//                       --- Test ZPBSVX ---
							if !prefac {
								golapack.Zlaset('F', toPtr(kd+1), &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), afac.CMatrix(ldab, opts), &ldab)
							}
							golapack.Zlaset('F', &n, nrhs, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), x.CMatrix(lda, opts), &lda)
							if iequed > 1 && n > 0 {
								//                          Equilibrate the matrix if FACT='F' and
								//                          EQUED='Y'
								golapack.Zlaqhb(uplo, &n, &kd, a.CMatrix(ldab, opts), &ldab, s, &scond, &amax, &equed)
							}

							//                       Solve the system and compute the condition
							//                       number and error bounds using ZPBSVX.
							*srnamt = "ZPBSVX"
							golapack.Zpbsvx(fact, uplo, &n, &kd, nrhs, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab, &equed, s, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)), work, rwork.Off(2*(*nrhs)), &info)

							//                       Check the error code from ZPBSVX.
							if info != izero {
								t.Fail()
								Alaerh(path, []byte("ZPBSVX"), &info, &izero, []byte{fact, uplo}, &n, &n, &kd, &kd, nrhs, &imat, &nfail, &nerrs)
								goto label60
							}

							if info == 0 {
								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									Zpbt01(uplo, &n, &kd, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab, rwork.Off(2*(*nrhs)), result.GetPtr(0))
									k1 = 1
								} else {
									k1 = 2
								}

								//                          Compute residual of the computed solution.
								golapack.Zlacpy('F', &n, nrhs, bsav.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
								Zpbt02(uplo, &n, &kd, nrhs, asav.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork.Off(2*(*nrhs)), result.GetPtr(1))

								//                          Check solution from generated exact solution.
								if nofact || (prefac && equed == 'N') {
									Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
								} else {
									Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &roldc, result.GetPtr(2))
								}

								//                          Check the error bounds from iterative
								//                          refinement.
								Zpbt05(uplo, &n, &kd, nrhs, asav.CMatrix(ldab, opts), &ldab, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)), result.Off(3))
							} else {
								k1 = 6
							}

							//                       Compare RCOND from ZPBSVX with the computed
							//                       value in RCONDC.
							result.Set(5, Dget06(&rcond, &rcondc))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = k1; k <= 6; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Aladhd(path)
									}
									if prefac {
										fmt.Printf(" %s( '%c', '%c', %5d, %5d, ... ), EQUED='%c', _type %1d, test(%1d)=%12.5f\n", "ZPBSVX", fact, uplo, n, kd, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s( '%c', '%c', %5d, %5d, ... ), _type %1d, test(%1d)=%12.5f\n", "ZPBSVX", fact, uplo, n, kd, imat, k, result.Get(k-1))
									}
									nfail = nfail + 1
								}
							}
							nrun = nrun + 7 - k1
						label60:
						}
					}
				label80:
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
