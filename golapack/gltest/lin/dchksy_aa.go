package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// DchksyAa tests DSYTRF_AA, -TRS_AA.
func DchksyAa(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo, xtype byte
	var anorm, cndnum, zero float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	ntypes = 10

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Test path
	path := []byte("DSA")

	//     Path to generate matrices
	matpath := []byte("DSY")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrsy(path, t)
	}
	(*infot) = 0

	//     Set the minimum block size for which the block routine should
	//     be used, which will be later returned by ILAENV
	Xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		if n > (*nmax) {
			nfail = nfail + 1
			fmt.Printf(" Invalid input value: %4s=%6d; must be <=%6d\n", "M ", n, *nmax)
			continue
		}
		lda = max(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		izero = 0

		//        Do for each value of matrix _type IMAT
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

					//                    Skip all tests for this generated matrix
					goto label160
				}

				//              For matrix types 3-6, zero one or more rows and
				//              columns of the matrix to test that INFO is returned
				//              correctly.
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
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = min(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
								ioff = ioff + lda
							}
							izero = 1
						} else {
							//                       Set the last IZERO rows and columns to zero.
							ioff = 0
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
				//
				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= (*nnb); inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					*srnamt = "DSYTRF_AA"
					lwork = max(1, n*nb+n)
					golapack.DsytrfAa(uplo, &n, afac.Matrix(lda, opts), &lda, iwork, ainv, &lwork, &info)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					//
					//c                  IF( IZERO.GT.0 ) THEN
					//c                     J = 1
					//c                     K = IZERO
					//c  100                CONTINUE
					//c                     IF( J.EQ.K ) THEN
					//c                        K = IWORK( J )
					//c                     ELSE IF( IWORK( J ).EQ.K ) THEN
					//c                        K = J
					//c                     END IF
					//c                     IF( J.LT.K ) THEN
					//c                        J = J + 1
					//c                        GO TO 100
					//c                     END IF
					//c                  ELSE
					k = 0
					//c                  END IF
					//
					//                 Check error code from DSYTRF and handle error.
					if info != k {
						Alaerh(path, []byte("DSYTRF_AA"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					Dsyt01Aa(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, ainv.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))
					nt = 1

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
							fmt.Printf(" UPLO = '%c', N =%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + nt

					//                 Skip solver test if INFO is not 0.
					if info != 0 {
						goto label140
					}

					//                 Do for each value of NRHS in NSVAL.
					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]

						//+    TEST 2 (Using TRS)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of NRHS random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "DLARHS"
						Dlarhs(matpath, &xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)
						//
						*srnamt = "DSYTRS_AA"
						lwork = max(1, 3*n-2)
						golapack.DsytrsAa(uplo, &n, &nrhs, afac.Matrix(lda, opts), &lda, iwork, x.Matrix(lda, opts), &lda, work, &lwork, &info)

						//                    Check error code from DSYTRS and handle error.
						if info != 0 {
							if izero == 0 {
								Alaerh(path, []byte("DSYTRS_AA"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}
						} else {
							golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)

							//                       Compute the residual for the solution
							Dpot02(uplo, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(1))

							//
							//                       Print information about the tests that did not pass
							//                       the threshold.
							for k = 2; k <= 2; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									t.Fail()
									fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
						}
						nrun = nrun + 1

						//                 End do for each value of NRHS in NSVAL.
					}
				label140:
				}
			label160:
			}
		label170:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 1320
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
