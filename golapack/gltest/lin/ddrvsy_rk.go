package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// DdrvsyRk tests the driver routines DSYSV_RK.
func DdrvsyRk(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, afac, e, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo, xtype byte
	var ainvnm, anorm, cndnum, one, rcondc, zero float64
	var i, i1, i2, ifact, imat, in, info, ioff, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nbmin, nerrs, nfact, nfail, nimat, nrun, nt, ntypes int

	facts := make([]byte, 2)
	uplos := make([]byte, 2)
	result := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 10
	nfact = 2

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], facts[0], facts[1] = 'U', 'L', 'F', 'N'

	//     Initialize constants and the random number seed.
	//
	//     Test path
	path := []byte("DSK")

	//     Path to generate matrices
	matpath := []byte("DSY")

	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	lwork = max(2*(*nmax), (*nmax)*(*nrhs))

	//     Test the error exits
	if *tsterr {
		Derrvx(path, t)
	}
	(*infot) = 0

	//     Set the block size and minimum block size for which the block
	//     routine should be used, which will be later returned by ILAENV.
	nb = 1
	nbmin = 2
	Xlaenv(1, nb)
	Xlaenv(2, nbmin)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = max(n, 1)
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

				//              Begin generate the test matrix A.
				//
				//              Set up parameters with DLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				Dlatb4(matpath, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				//              Generate a matrix with DLATMS.
				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS and handle error.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)

					//                 Skip all tests for this generated matrix
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
						ioff = 0
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i2 = min(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i1 = max(j, izero)
								for i = i1; i <= n; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
						}
					}
				} else {
					izero = 0
				}

				//              End generate the test matrix A.
				for ifact = 1; ifact <= nfact; ifact++ {
					//                 Do first for FACT = 'F', then for other values.
					// fact = facts[ifact-1]

					//                 Compute the condition number
					if zerot {
						if ifact == 1 {
							goto label150
						}
						rcondc = zero

					} else if ifact == 1 {
						//                    Compute the 1-norm of A.
						anorm = golapack.Dlansy('1', uplo, &n, a.Matrix(lda, opts), &lda, rwork)

						//                    Factor the matrix A.
						golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
						golapack.DsytrfRk(uplo, &n, afac.Matrix(lda, opts), &lda, e, iwork, work, &lwork, &info)

						//                    Compute inv(A) and take its norm.
						golapack.Dlacpy(uplo, &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
						lwork = (n + nb + 1) * (nb + 3)

						//                    We need to copute the invesrse to compute
						//                    RCONDC that is used later in TEST3.
						golapack.Dsytri3(uplo, &n, ainv.Matrix(lda, opts), &lda, e, iwork, work, &lwork, &info)
						ainvnm = golapack.Dlansy('1', uplo, &n, ainv.Matrix(lda, opts), &lda, rwork)

						//                    Compute the 1-norm condition number of A.
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}
					}

					//                 Form an exact solution and set the right hand side.
					*srnamt = "DLARHS"
					Dlarhs(matpath, &xtype, uplo, ' ', &n, &n, &kl, &ku, nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
					xtype = 'C'

					//                 --- Test DSYSV_RK  ---
					if ifact == 2 {
						golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						//                    Factor the matrix and solve the system using
						//                    DSYSV_RK.
						*srnamt = "DSYSV_RK"
						golapack.DsysvRk(uplo, &n, nrhs, afac.Matrix(lda, opts), &lda, e, iwork, x.Matrix(lda, opts), &lda, work, &lwork, &info)

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

						//                    Check error code from DSYSV_RK and handle error.
						if info != k {
							Alaerh(path, []byte("DSYSV_RK"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), nrhs, &imat, &nfail, &nerrs)
							goto label120
						} else if info != 0 {
							goto label120
						}

						//+    TEST 1      Reconstruct matrix from factors and compute
						//                 residual.
						Dsyt013(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, e, iwork, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))

						//+    TEST 2      Compute residual of the computed solution.
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
						Dpot02(uplo, &n, nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

						//+    TEST 3
						//                 Check solution from generated exact solution.
						Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
						nt = 3

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 1; k <= nt; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								fmt.Printf(" %s, UPLO='%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", "DSYSV_RK", uplo, n, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt
					label120:
					}

				label150:
				}

			label160:
			}
		label170:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 222
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
