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

// zdrvpb tests the driver routines Zpbsvand -SVX.
func zdrvpb(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, afac, asav, b, bsav, x, xact *mat.CVector, s *mat.Vector, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
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

	one = 1.0
	zero = 0.0
	ntypes = 8
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1], facts[2], equeds[0], equeds[1] = 'F', 'N', 'E', 'N', 'Y'

	//     Initialize constants and the random number seed.
	path := "Zpb"
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
		nkd = max(1, min(n, 4))
		nimat = ntypes
		if n == 0 {
			nimat = 1
		}

		kdval[1] = n + (n+1)/4
		kdval[2] = (3*n - 1) / 4
		kdval[3] = (n + 1) / 4

		for ikd = 1; ikd <= nkd; ikd++ {
			//           Do for kd= 0, (5*N+1)/4, (3N-1)/4, and (N+1)/4. This order
			//           makes it easier to skip redundant values for small values
			//           of N.
			kd = kdval[ikd-1]
			ldab = kd + 1

			//           Do first for uplo='U', then for uplo='L'
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
						//                    Set up parameters with ZLATB4 and generate a test
						//                    matrix with Zlatms.
						_type, _, _, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

						*srnamt = "Zlatms"
						if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kd, kd, packit, a.CMatrixOff(koff-1, ldab, opts), work); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
							goto label80
						}
					} else if izero > 0 {
						//                    Use the same matrix for types 3 and 4 as for _type
						//                    2 by copying back the zeroed out column,
						iw = 2*lda + 1
						if uplo == Upper {
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

						if uplo == Upper {
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
					if uplo == Upper {
						zlaipd(n, a.Off(kd), ldab, 0)
					} else {
						zlaipd(n, a.Off(0), ldab, 0)
					}

					//                 Save a copy of the matrix A in ASAV.
					golapack.Zlacpy(Full, kd+1, n, a.CMatrix(ldab, opts), asav.CMatrix(ldab, opts))

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
								//                          with the value returned by Zpbsvx (FACT =
								//                          'N' reuses the condition number from the
								//                          previous iteration with FACT = 'F').
								golapack.Zlacpy(Full, kd+1, n, asav.CMatrix(ldab, opts), afac.CMatrix(ldab, opts))
								if equil || iequed > 1 {
									//                             Compute row and column scale factors to
									//                             equilibrate the matrix A.
									if scond, amax, info, err = golapack.Zpbequ(uplo, n, kd, afac.CMatrix(ldab, opts), s); err != nil {
										panic(err)
									}
									if info == 0 && n > 0 {
										if iequed > 1 {
											scond = zero
										}

										//                                Equilibrate the matrix.
										equed = golapack.Zlaqhb(uplo, n, kd, afac.CMatrix(ldab, opts), s, scond, amax)
									}
								}

								//                          Save the condition number of the
								//                          non-equilibrated system for use in ZGET04.
								if equil {
									roldc = rcondc
								}

								//                          Compute the 1-norm of A.
								anorm = golapack.Zlanhb('1', uplo, n, kd, afac.CMatrix(ldab, opts), rwork)

								//                          Factor the matrix A.
								if info, err = golapack.Zpbtrf(uplo, n, kd, afac.CMatrix(ldab, opts)); err != nil {
									panic(err)
								}

								//                          Form the inverse of A.
								golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), a.CMatrix(lda, opts))
								*srnamt = "Zpbtrs"
								if err = golapack.Zpbtrs(uplo, n, kd, n, afac.CMatrix(ldab, opts), a.CMatrix(lda, opts)); err != nil {
									panic(err)
								}

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Zlange('1', n, n, a.CMatrix(lda, opts), rwork)
								if anorm <= zero || ainvnm <= zero {
									rcondc = one
								} else {
									rcondc = (one / anorm) / ainvnm
								}
							}

							//                       Restore the matrix A.
							golapack.Zlacpy(Full, kd+1, n, asav.CMatrix(ldab, opts), a.CMatrix(ldab, opts))

							//                       Form an exact solution and set the right hand
							//                       side.
							*srnamt = "zlarhs"
							if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kd, kd, nrhs, a.CMatrix(ldab, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), bsav.CMatrix(lda, opts))

							if nofact {
								//                          --- Test Zpbsv ---
								//
								//                          Compute the L*L' or U'*U factorization of the
								//                          matrix and solve the system.
								golapack.Zlacpy(Full, kd+1, n, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts))
								golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

								*srnamt = "Zpbsv"
								if info, err = golapack.Zpbsv(uplo, n, kd, nrhs, afac.CMatrix(ldab, opts), x.CMatrix(lda, opts)); err != nil {
									panic(err)
								}

								//                          Check error code from Zpbsv.
								if info != izero {
									t.Fail()
									nerrs = alaerh(path, "Zpbsv", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
									goto label40
								} else if info != 0 {
									goto label40
								}

								//                          Reconstruct matrix from factors and compute
								//                          residual.
								*result.GetPtr(0) = zpbt01(uplo, n, kd, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts), rwork)

								//                          Compute residual of the computed solution.
								golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
								*result.GetPtr(1) = zpbt02(uplo, n, kd, nrhs, a.CMatrix(ldab, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

								//                          Check solution from generated exact solution.
								*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
								nt = 3

								//                          Print information about the tests that did
								//                          not pass the threshold.
								for k = 1; k <= nt; k++ {
									if result.Get(k-1) >= thresh {
										t.Fail()
										if nfail == 0 && nerrs == 0 {
											aladhd(path)
										}
										fmt.Printf(" %s, uplo='%c', n=%5d, kd=%5d, _type %1d, test(%1d)=%12.5f\n", "Zpbsv", uplo, n, kd, imat, k, result.Get(k-1))
										nfail++
									}
								}
								nrun = nrun + nt
							label40:
							}

							//                       --- Test Zpbsvx ---
							if !prefac {
								golapack.Zlaset(Full, kd+1, n, complex(zero, 0), complex(zero, 0), afac.CMatrix(ldab, opts))
							}
							golapack.Zlaset(Full, n, nrhs, complex(zero, 0), complex(zero, 0), x.CMatrix(lda, opts))
							if iequed > 1 && n > 0 {
								//                          Equilibrate the matrix if fact='F' and
								//                          EQUED='Y'
								equed = golapack.Zlaqhb(uplo, n, kd, a.CMatrix(ldab, opts), s, scond, amax)
							}

							//                       Solve the system and compute the condition
							//                       number and error bounds using Zpbsvx.
							*srnamt = "Zpbsvx"
							if equed, rcond, info, err = golapack.Zpbsvx(fact, uplo, n, kd, nrhs, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts), equed, s, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
								panic(err)
							}

							//                       Check the error code from Zpbsvx.
							if info != izero {
								t.Fail()
								nerrs = alaerh(path, "Zpbsvx", info, 0, []byte{fact, uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
								goto label60
							}

							if info == 0 {
								if !prefac {
									//                             Reconstruct matrix from factors and
									//                             compute residual.
									*result.GetPtr(0) = zpbt01(uplo, n, kd, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts), rwork.Off(2*nrhs))
									k1 = 1
								} else {
									k1 = 2
								}

								//                          Compute residual of the computed solution.
								golapack.Zlacpy(Full, n, nrhs, bsav.CMatrix(lda, opts), work.CMatrix(lda, opts))
								*result.GetPtr(1) = zpbt02(uplo, n, kd, nrhs, asav.CMatrix(ldab, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork.Off(2*nrhs))

								//                          Check solution from generated exact solution.
								if nofact || (prefac && equed == 'N') {
									*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
								} else {
									*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), roldc)
								}

								//                          Check the error bounds from iterative
								//                          refinement.
								zpbt05(uplo, n, kd, nrhs, asav.CMatrix(ldab, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
							} else {
								k1 = 6
							}

							//                       Compare RCOND from Zpbsvx with the computed
							//                       value in RCONDC.
							result.Set(5, dget06(rcond, rcondc))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = k1; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										aladhd(path)
									}
									if prefac {
										fmt.Printf(" %s( '%c', %s, %5d, %5d, ... ), equed='%c', _type %1d, test(%1d)=%12.5f\n", "Zpbsvx", fact, uplo, n, kd, equed, imat, k, result.Get(k-1))
									} else {
										fmt.Printf(" %s( '%c', %s, %5d, %5d, ... ), _type %1d, test(%1d)=%12.5f\n", "Zpbsvx", fact, uplo, n, kd, imat, k, result.Get(k-1))
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

	//     Print a summary of the results.
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
