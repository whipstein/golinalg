package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvsy tests the driver routines Zsysvand -SVX.
func zdrvsy(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var zerot bool
	var dist, fact, _type, xtype byte
	var uplo mat.MatUplo
	var ainvnm, anorm, cndnum, one, rcond, rcondc, zero float64
	var i, i1, i2, ifact, imat, in, info, ioff, izero, j, k, k1, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntypes int
	var err error

	facts := make([]byte, 2)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	one = 1.0
	zero = 0.0
	ntypes = 11
	nfact = 2
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	facts[0], facts[1] = 'F', 'N'

	//     Initialize constants and the random number seed.
	path := "Zsy"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	lwork = max(2*nmax, nmax*nrhs)

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
				goto label170
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label170
			}

			//           Do first for uplo='U', then for uplo='L'
			for _, uplo = range mat.IterMatUplo(false) {

				if imat != ntypes {
					//                 Set up parameters with ZLATB4 and generate a test
					//                 matrix with Zlatms.
					_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

					*srnamt = "Zlatms"
					if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.CMatrix(lda, opts), work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						goto label160
					}

					//                 For types 3-6, zero one or more rows and columns of
					//                 the matrix to test that INFO is returned correctly.
					if zerot {
						if imat == 3 {
							izero = 1
						} else if imat == 4 {
							izero = n
						} else {
							izero = n/2 + 1
						}

						if imat < 6 {
							//                       Set row and column IZERO to zero.
							if uplo == Upper {
								ioff = (izero - 1) * lda
								for i = 1; i <= izero-1; i++ {
									a.SetRe(ioff+i-1, zero)
								}
								ioff = ioff + izero
								for i = izero; i <= n; i++ {
									a.SetRe(ioff-1, zero)
									ioff = ioff + lda
								}
							} else {
								ioff = izero
								for i = 1; i <= izero-1; i++ {
									a.SetRe(ioff-1, zero)
									ioff = ioff + lda
								}
								ioff = ioff - izero
								for i = izero; i <= n; i++ {
									a.SetRe(ioff+i-1, zero)
								}
							}
						} else {
							if uplo == Upper {
								//                          Set the first IZERO rows to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i2 = min(j, izero)
									for i = 1; i <= i2; i++ {
										a.SetRe(ioff+i-1, zero)
									}
									ioff = ioff + lda
								}
							} else {
								//                          Set the last IZERO rows to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i1 = max(j, izero)
									for i = i1; i <= n; i++ {
										a.SetRe(ioff+i-1, zero)
									}
									ioff = ioff + lda
								}
							}
						}
					} else {
						izero = 0
					}
				} else {
					//                 IMAT = NTYPES:  Use a special block diagonal matrix to
					//                 test alternate code for the 2-by-2 blocks.
					zlatsy(uplo, n, a.CMatrix(lda, opts), &iseed)
				}

				for ifact = 1; ifact <= nfact; ifact++ {
					//                 Do first for FACT = 'F', then for other values.
					fact = facts[ifact-1]

					//                 Compute the condition number for comparison with
					//                 the value returned by Zsysvx.
					if zerot {
						if ifact == 1 {
							goto label150
						}
						rcondc = zero

					} else if ifact == 1 {
						//                    Compute the 1-norm of A.
						anorm = golapack.Zlansy('1', uplo, n, a.CMatrix(lda, opts), rwork)

						//                    Factor the matrix A.
						golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))
						if info, err = golapack.Zsytrf(uplo, n, afac.CMatrix(lda, opts), &iwork, work, lwork); err != nil {
							panic(err)
						}

						//                    Compute inv(A) and take its norm.
						golapack.Zlacpy(uplo, n, n, afac.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
						lwork = (n + nb + 1) * (nb + 3)
						if info, err = golapack.Zsytri2(uplo, n, ainv.CMatrix(lda, opts), &iwork, work, lwork); err != nil {
							panic(err)
						}
						ainvnm = golapack.Zlansy('1', uplo, n, ainv.CMatrix(lda, opts), rwork)

						//                    Compute the 1-norm condition number of A.
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}
					}

					//                 Form an exact solution and set the right hand side.
					*srnamt = "zlarhs"
					if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
						panic(err)
					}
					xtype = 'C'

					//                 --- Test Zsysv ---
					if ifact == 2 {
						golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						//                    Factor the matrix and solve the system using ZSYSV.
						*srnamt = "Zsysv"
						info, err = golapack.Zsysv(uplo, n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts), work, lwork)

						//                    Adjust the expected value of INFO to account for
						//                    pivoting.
						k = izero
						if k > 0 {
						label100:
							;
							if iwork[k-1] < 0 {
								if iwork[k-1] != -k {
									k = -iwork[k-1]
									goto label100
								}
							} else if iwork[k-1] != k {
								k = iwork[k-1]
								goto label100
							}
						}

						//                    Check error code from Zsysv.
						if err != nil || info != k {
							t.Fail()
							nerrs = alaerh(path, "Zsysv", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							goto label120
						} else if info != 0 {
							goto label120
						}

						//                    Reconstruct matrix from factors and compute
						//                    residual.
						*result.GetPtr(0) = zsyt01(uplo, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, ainv.CMatrix(lda, opts), rwork)

						//                    Compute residual of the computed solution.
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
						*result.GetPtr(1) = zsyt02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

						//                    Check solution from generated exact solution.
						*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
						nt = 3

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 1; k <= nt; k++ {
							if result.Get(k-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								fmt.Printf(" %s, uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", "Zsysv", uplo, n, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun = nrun + nt
					label120:
					}

					//                 --- Test Zsysvx ---
					if ifact == 2 {
						golapack.Zlaset(uplo, n, n, complex(zero, 0), complex(zero, 0), afac.CMatrix(lda, opts))
					}
					golapack.Zlaset(Full, n, nrhs, complex(zero, 0), complex(zero, 0), x.CMatrix(lda, opts))

					//                 Solve the system and compute the condition number and
					//                 error bounds using Zsysvx.
					*srnamt = "Zsysvx"
					rcond, info, err = golapack.Zsysvx(fact, uplo, n, nrhs, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, lwork, rwork.Off(2*nrhs))

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					k = izero
					if k > 0 {
					label130:
						;
						if iwork[k-1] < 0 {
							if iwork[k-1] != -k {
								k = -iwork[k-1]
								goto label130
							}
						} else if iwork[k-1] != k {
							k = iwork[k-1]
							goto label130
						}
					}

					//                 Check the error code from Zsysvx.
					if err != nil || info != k {
						t.Fail()
						nerrs = alaerh(path, "Zsysvx", info, k, []byte{fact, uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						goto label150
					}

					if info == 0 {
						if ifact >= 2 {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							*result.GetPtr(0) = zsyt01(uplo, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, ainv.CMatrix(lda, opts), rwork.Off(2*nrhs))
							k1 = 1
						} else {
							k1 = 2
						}

						//                    Compute residual of the computed solution.
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
						*result.GetPtr(1) = zsyt02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork.Off(2*nrhs))

						//                    Check solution from generated exact solution.
						*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

						//                    Check the error bounds from iterative refinement.
						zpot05(uplo, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
					} else {
						k1 = 6
					}

					//                 Compare RCOND from Zsysvx with the computed value
					//                 in RCONDC.
					result.Set(5, dget06(rcond, rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= 6; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								aladhd(path)
							}
							fmt.Printf(" %s, fact='%c', uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", "Zsysvx", fact, uplo, n, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 7 - k1

				label150:
				}

			label160:
			}
		label170:
		}
	}

	//     Print a summary of the results.
	alasvm(path, nfail, nrun, nerrs)
}
