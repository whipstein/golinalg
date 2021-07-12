package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkheaa2stage tests ZHETRF_AA_2STAGE, -TRS_AA_2STAGE.
func Zchkheaa2stage(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo, xtype byte
	var czero complex128
	var anorm, cndnum float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	czero = (0.0 + 0.0*1i)
	ntypes = 10
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	//
	//     Test path
	path := []byte("ZH2")

	//     Path to generate matrices
	matpath := []byte("ZHE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrhe(path, t)
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
			goto label180
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
				//              Set up parameters with ZLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				Zlatb4(matpath, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				//              Generate a matrix with ZLATMS.
				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, uplo, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS and handle error.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)

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
								a.Set(ioff+i-1, czero)
							}
							ioff = ioff + izero
							for i = izero; i <= n; i++ {
								a.Set(ioff-1, czero)
								ioff = ioff + lda
							}
						} else {
							ioff = izero
							for i = 1; i <= izero-1; i++ {
								a.Set(ioff-1, czero)
								ioff = ioff + lda
							}
							ioff = ioff - izero
							for i = izero; i <= n; i++ {
								a.Set(ioff+i-1, czero)
							}
						}
					} else {
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							ioff = 0
							for j = 1; j <= n; j++ {
								i2 = min(j, izero)
								for i = 1; i <= i2; i++ {
									a.Set(ioff+i-1, czero)
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
									a.Set(ioff+i-1, czero)
								}
								ioff = ioff + lda
							}
						}
					}
				} else {
					izero = 0
				}

				//              End generate test matrix A.
				//
				//
				//              Set the imaginary part of the diagonals.
				Zlaipd(&n, a, toPtr(lda+1), func() *int { y := 0; return &y }())

				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= (*nnb); inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, afac.CMatrix(lda, opts), &lda)

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					*srnamt = "ZHETRF_AA_2STAGE"
					lwork = min(n*nb, 3*(*nmax)*(*nmax))
					golapack.Zhetrfaa2stage(uplo, &n, afac.CMatrix(lda, opts), &lda, ainv, toPtr((3*nb+1)*n), iwork, toSlice(iwork, 1+n-1), work, &lwork, &info)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					if izero > 0 {
						j = 1
						k = izero
					label100:
						;
						if j == k {
							k = (*iwork)[j-1]
						} else if (*iwork)[j-1] == k {
							k = j
						}
						if j < k {
							j = j + 1
							goto label100
						}
					} else {
						k = 0
					}

					//                 Check error code from CHETRF and handle error.
					if info != k {
						t.Fail()
						Alaerh(path, []byte("ZHETRF_AA_2STAGE"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					//
					//                 NEED TO CREATE ZHET01_AA_2STAGE
					//                  CALL ZHET01_AA( UPLO, N, A, LDA, AFAC, LDA, IWORK,
					//     $                            AINV, LDA, RWORK, RESULT( 1 ) )
					//                  NT = 1
					nt = 0

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
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
						*srnamt = "ZLARHS"
						Zlarhs(matpath, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
						golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZHETRS_AA_2STAGE"
						lwork = max(1, 3*n-2)
						golapack.Zhetrsaa2stage(uplo, &n, &nrhs, afac.CMatrix(lda, opts), &lda, ainv, toPtr((3*nb+1)*n), iwork, toSlice(iwork, 1+n-1), x.CMatrix(lda, opts), &lda, &info)

						//                    Check error code from ZHETRS and handle error.
						if info != 0 {
							if izero == 0 {
								t.Fail()
								Alaerh(path, []byte("ZHETRS_AA_2STAGE"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}
						} else {

							golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)

							//                       Compute the residual for the solution
							Zpot02(uplo, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

							//                       Print information about the tests that did not pass
							//                       the threshold.
							for k = 2; k <= 2; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
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
	label180:
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
