package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Zdrvhe tests the driver routines ZHESV and -SVX.
func Zdrvhe(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, fact, _type, uplo, xtype byte
	var ainvnm, anorm, cndnum, one, rcond, rcondc, zero float64
	var i, i1, i2, ifact, imat, in, info, ioff, iuplo, izero, j, k, k1, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntypes int

	facts := make([]byte, 2)
	uplos := make([]byte, 2)
	result := vf(6)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 10
	nfact = 2
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], facts[0], facts[1] = 'U', 'L', 'F', 'N'

	//     Initialize constants and the random number seed.
	path := []byte("ZHE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	lwork = maxint(2*(*nmax), (*nmax)*(*nrhs))

	//     Test the error exits
	if *tsterr {
		Zerrvx(path, t)
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
		lda = maxint(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label170
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label170
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with ZLATMS.
				Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label160
				}

				//              For types 3-6, zero one or more rows and columns of the
				//              matrix to test that INFO is returned correctly.
				if zerot {
					if imat == 3 {
						izero = 1
					} else if imat == 4 {
						izero = n
					} else {
						izero = n/2 + 1
					}

					if imat < 6 {
						//                    Set row and column IZERO to zero.
						if iuplo == 1 {
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
						ioff = 0
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i2 = minint(j, izero)
								for i = 1; i <= i2; i++ {
									a.SetRe(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i1 = maxint(j, izero)
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

				//              Set the imaginary part of the diagonals.
				Zlaipd(&n, a, toPtr(lda+1), func() *int { y := 0; return &y }())

				for ifact = 1; ifact <= nfact; ifact++ {
					//                 Do first for FACT = 'F', then for other values.
					fact = facts[ifact-1]

					//                 Compute the condition number for comparison with
					//                 the value returned by ZHESVX.
					if zerot {
						if ifact == 1 {
							goto label150
						}
						rcondc = zero

					} else if ifact == 1 {
						//                    Compute the 1-norm of A.
						anorm = golapack.Zlanhe('1', uplo, &n, a.CMatrix(lda, opts), &lda, rwork)

						//                    Factor the matrix A.
						golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)
						golapack.Zhetrf(uplo, &n, afac.CMatrix(lda, opts), &lda, iwork, work, &lwork, &info)

						//                    Compute inv(A) and take its norm.
						golapack.Zlacpy(uplo, &n, &n, afac.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
						lwork = (n + nb + 1) * (nb + 3)
						golapack.Zhetri2(uplo, &n, ainv.CMatrix(lda, opts), &lda, iwork, work, &lwork, &info)
						ainvnm = golapack.Zlanhe('1', uplo, &n, ainv.CMatrix(lda, opts), &lda, rwork)

						//                    Compute the 1-norm condition number of A.
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}
					}

					//                 Form an exact solution and set the right hand side.
					*srnamt = "ZLARHS"
					Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
					xtype = 'C'

					//                 --- Test ZHESV  ---
					if ifact == 2 {
						golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						//                    Factor the matrix and solve the system using ZHESV.
						*srnamt = "ZHESV "
						golapack.Zhesv(uplo, &n, nrhs, afac.CMatrix(lda, opts), &lda, iwork, x.CMatrix(lda, opts), &lda, work, &lwork, &info)

						//                    Adjust the expected value of INFO to account for
						//                    pivoting.
						k = izero
						if k > 0 {
						label100:
							;
							if (*iwork)[k-1] < 0 {
								if (*iwork)[k-1] != -k {
									k = -(*iwork)[k-1]
									goto label100
								}
							} else if (*iwork)[k-1] != k {
								k = (*iwork)[k-1]
								goto label100
							}
						}

						//                    Check error code from ZHESV .
						if info != k {
							t.Fail()
							Alaerh(path, []byte("ZHESV "), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label120
						} else if info != 0 {
							goto label120
						}

						//                    Reconstruct matrix from factors and compute
						//                    residual.
						Zhet01(uplo, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, ainv.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))

						//                    Compute residual of the computed solution.
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
						Zpot02(uplo, &n, nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
						nt = 3

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 1; k <= nt; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								fmt.Printf(" %s, UPLO='%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", "ZHESV ", uplo, n, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt
					label120:
					}

					//                 --- Test ZHESVX ---
					if ifact == 2 {
						golapack.Zlaset(uplo, &n, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), afac.CMatrix(lda, opts), &lda)
					}
					golapack.Zlaset('F', &n, nrhs, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), x.CMatrix(lda, opts), &lda)

					//                 Solve the system and compute the condition number and
					//                 error bounds using ZHESVX.
					*srnamt = "ZHESVX"
					golapack.Zhesvx(fact, uplo, &n, nrhs, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, &lwork, rwork.Off(2*(*nrhs)+1-1), &info)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					k = izero
					if k > 0 {
					label130:
						;
						if (*iwork)[k-1] < 0 {
							if (*iwork)[k-1] != -k {
								k = -(*iwork)[k-1]
								goto label130
							}
						} else if (*iwork)[k-1] != k {
							k = (*iwork)[k-1]
							goto label130
						}
					}

					//                 Check the error code from ZHESVX.
					if info != k {
						t.Fail()
						Alaerh(path, []byte("ZHESVX"), &info, &k, []byte{fact, uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
						goto label150
					}

					if info == 0 {
						if ifact >= 2 {
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Zhet01(uplo, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda, iwork, ainv.CMatrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(0))
							k1 = 1
						} else {
							k1 = 2
						}

						//                    Compute residual of the computed solution.
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
						Zpot02(uplo, &n, nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork.Off(2*(*nrhs)+1-1), result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

						//                    Check the error bounds from iterative refinement.
						Zpot05(uplo, &n, nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)+1-1), result.Off(3))
					} else {
						k1 = 6
					}

					//                 Compare RCOND from ZHESVX with the computed value
					//                 in RCONDC.
					result.Set(5, Dget06(&rcond, &rcondc))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= 6; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Aladhd(path)
							}
							fmt.Printf(" %s, FACT='%c', UPLO='%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", "ZHESVX", fact, uplo, n, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 7 - k1

				label150:
				}

			label160:
			}
		label170:
		}
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}