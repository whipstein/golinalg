package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchksyAa2stage tests ZsytrfAa2stage, -TRS_AA_2STAGE.
func zchksyAa2stage(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var zerot bool
	var dist, _type, xtype byte
	var uplo mat.MatUplo
	var czero complex128
	var anorm, cndnum float64
	var i, i1, i2, imat, in, inb, info, ioff, irhs, izero, j, k, kl, ku, lda, lwork, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	czero = (0.0 + 0.0*1i)
	ntypes = 10
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	//
	//     Test path
	path := "Zs2"
	alasumStart(path)

	//     Path to generate matrices
	matpath := "Zsy"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrsy(path, t)
	}
	(*infot) = 0

	//     Set the minimum block size for which the block routine should
	//     be used, which will be later returned by ILAENV
	xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= nn; in++ {
		n = nval[in-1]
		if n > nmax {
			nfail++
			fmt.Printf(" Invalid input value: %4s=%6d; must be <=%6d\n", "M ", n, nmax)
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

				//              Begin generate the test matrix A.
				//
				//
				//              Set up parameters with ZLATB4 for the matrix generator
				//              based on the _type of matrix to be generated.
				_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(matpath, imat, n, n)

				//              Generate a matrix with Zlatms.
				*srnamt = "Zlatms"
				if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, uplo.Byte(), a.CMatrix(lda, opts), work); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)

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
						if uplo == Upper {
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
						if uplo == Upper {
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

				//              End generate the test matrix A.
				//
				//              Do for each value of NB in NBVAL
				for inb = 1; inb <= nnb; inb++ {
					//                 Set the optimal blocksize, which will be later
					//                 returned by ILAENV.
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//                 Copy the test matrix A into matrix AFAC which
					//                 will be factorized in place. This is needed to
					//                 preserve the test matrix A for subsequent tests.
					golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))

					//                 Compute the L*D*L**T or U*D*U**T factorization of the
					//                 matrix. IWORK stores details of the interchanges and
					//                 the block structure of D. AINV is a work array for
					//                 block factorization, LWORK is the length of AINV.
					*srnamt = "ZsytrfAa2stage"
					lwork = min(n*nb, 3*nmax*nmax)
					info, err = golapack.ZsytrfAa2stage(uplo, n, afac.CMatrix(lda, opts), ainv, (3*nb+1)*n, &iwork, toSlice(&iwork, 1+n-1), work, lwork)

					//                 Adjust the expected value of INFO to account for
					//                 pivoting.
					if izero > 0 {
						j = 1
						k = izero
					label100:
						;
						if j == k {
							k = iwork[j-1]
						} else if iwork[j-1] == k {
							k = j
						}
						if j < k {
							j = j + 1
							goto label100
						}
					} else {
						k = 0
					}

					//                 Check error code from ZSYTRF and handle error.
					if err != nil || info != k {
						t.Fail()
						nerrs = alaerh(path, "ZsytrfAa2stage", info, k, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					//
					//c                  CALL ZSYT01_AA( UPLO, N, A, LDA, AFAC, LDA, IWORK,
					//c     $                            AINV, LDA, RWORK, RESULT( 1 ) )
					//c                  NT = 1
					nt = 0

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, nb=%4d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Skip solver test if INFO is not 0.
					if info != 0 {
						goto label140
					}

					//                 Do for each value of NRHS in NSVAL.
					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]

						//+    TEST 2 (Using TRS)
						//                 Solve and compute residual for  A * X = B.
						//
						//                    Choose a set of NRHS random solution vectors
						//                    stored in XACT and set up the right hand side B
						*srnamt = "zlarhs"
						if err = zlarhs(matpath, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "ZsytrsAa2stage"
						lwork = max(1, 3*n-2)
						if err = golapack.ZsytrsAa2stage(uplo, n, nrhs, afac.CMatrix(lda, opts), ainv, (3*nb+1)*n, &iwork, toSlice(&iwork, 1+n-1), x.CMatrix(lda, opts)); err != nil {
							if izero == 0 {
								t.Fail()
								nerrs = alaerh(path, "ZsytrsAa2stage", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}
						} else {
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))

							//                       Compute the residual for the solution
							*result.GetPtr(1) = zsyt02(uplo, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

							//
							//                       Print information about the tests that did not pass
							//                       the threshold.
							for k = 2; k <= 2; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" uplo=%s, n=%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
									nfail++
								}
							}
						}
						nrun++

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
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
