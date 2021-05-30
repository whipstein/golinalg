package lin

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Zchkhp tests ZHPTRF, -TRI, -TRS, -RFS, and -CON
func Zchkhp(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, packit, _type, uplo, xtype byte
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, i1, i2, imat, in, info, ioff, irhs, iuplo, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, npp, nrhs, nrun, nt, ntypes int

	uplos := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	zero = 0.0
	ntypes = 10
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("ZHP")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrsy(path, t)
	}
	(*infot) = 0

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		izero = 0
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label160
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label160
			}

			//           Do first for UPLO = 'U', then for UPLO = 'L'
			for iuplo = 1; iuplo <= 2; iuplo++ {
				uplo = uplos[iuplo-1]
				if uplo == 'U' {
					packit = 'C'
				} else {
					packit = 'R'
				}

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with ZLATMS.
				Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label150
				}

				//              For types 3-6, zero one or more rows and columns of
				//              the matrix to test that INFO is returned correctly.
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
						ioff = 0
						if iuplo == 1 {
							//                       Set the first IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i2 = minint(j, izero)
								for i = 1; i <= i2; i++ {
									a.SetRe(ioff+i-1, zero)
								}
								ioff = ioff + j
							}
						} else {
							//                       Set the last IZERO rows and columns to zero.
							for j = 1; j <= n; j++ {
								i1 = maxint(j, izero)
								for i = i1; i <= n; i++ {
									a.SetRe(ioff+i-1, zero)
								}
								ioff = ioff + n - j
							}
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

				//              Compute the L*D*L' or U*D*U' factorization of the matrix.
				npp = n * (n + 1) / 2
				goblas.Zcopy(&npp, a, func() *int { y := 1; return &y }(), afac, func() *int { y := 1; return &y }())
				*srnamt = "ZHPTRF"
				golapack.Zhptrf(uplo, &n, afac, iwork, &info)

				//              Adjust the expected value of INFO to account for
				//              pivoting.
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

				//              Check error code from ZHPTRF.
				if info != k {
					Alaerh(path, []byte("ZHPTRF"), &info, &k, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}
				if info != 0 {
					trfcon = true
				} else {
					trfcon = false
				}

				//+    TEST 1
				//              Reconstruct matrix from factors and compute residual.
				Zhpt01(uplo, &n, a, afac, iwork, ainv.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))
				nt = 1

				//+    TEST 2
				//              Form the inverse and compute the residual.
				if !trfcon {
					goblas.Zcopy(&npp, afac, func() *int { y := 1; return &y }(), ainv, func() *int { y := 1; return &y }())
					*srnamt = "ZHPTRI"
					golapack.Zhptri(uplo, &n, ainv, iwork, work, &info)

					//              Check error code from ZHPTRI.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZHPTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Zppt03(uplo, &n, a, ainv, work.CMatrix(lda, opts), &lda, rwork, &rcondc, result.GetPtr(1))
					nt = 2
				}

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = 1; k <= nt; k++ {
					if result.Get(k-1) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
				}
				nrun = nrun + nt

				//              Do only the condition estimate if INFO is not 0.
				if trfcon {
					rcondc = zero
					goto label140
				}

				for irhs = 1; irhs <= (*nns); irhs++ {
					nrhs = (*nsval)[irhs-1]

					//+    TEST 3
					//              Solve and compute residual for  A * X = B.
					*srnamt = "ZLARHS"
					Zlarhs(path, xtype, uplo, ' ', &n, &n, &kl, &ku, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
					xtype = 'C'
					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

					*srnamt = "ZHPTRS"
					golapack.Zhptrs(uplo, &n, &nrhs, afac, iwork, x.CMatrix(lda, opts), &lda, &info)

					//              Check error code from ZHPTRS.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZHPTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
					Zppt02(uplo, &n, &nrhs, a, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(2))

					//+    TEST 4
					//              Check solution from generated exact solution.
					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

					//+    TESTS 5, 6, and 7
					//              Use iterative refinement to improve the solution.
					*srnamt = "ZHPRFS"
					golapack.Zhprfs(uplo, &n, &nrhs, a, afac, iwork, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)
					//
					//              Check error code from ZHPRFS.
					//
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZHPRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(4))
					Zppt05(uplo, &n, &nrhs, a, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(5))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 3; k <= 7; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO = '%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 5
				}

				//+    TEST 8
				//              Get an estimate of RCOND = 1/CNDNUM.
			label140:
				;
				anorm = golapack.Zlanhp('1', uplo, &n, a, rwork)
				*srnamt = "ZHPCON"
				golapack.Zhpcon(uplo, &n, afac, iwork, &anorm, &rcond, work, &info)

				//              Check error code from ZHPCON.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZHPCON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				result.Set(7, Dget06(&rcond, &rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(7) >= (*thresh) {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						Alahd(path)
					}
					fmt.Printf(" UPLO = '%c', N =%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, 8, result.Get(7))
					nfail = nfail + 1
				}
				nrun = nrun + 1
			label150:
			}
		label160:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
