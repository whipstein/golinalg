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

// zchksp tests Zsptrf, -TRI, -TRS, -RFS, and -CON
func zchksp(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, packit, _type, xtype byte
	var uplo mat.MatUplo
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, i1, i2, imat, in, info, ioff, irhs, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, npp, nrhs, nrun, nt, ntypes int
	var err error

	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	zero = 0.0
	ntypes = 11
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Zsp"
	alasumStart(path)
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
				goto label160
			}

			//           Skip types 3, 4, 5, or 6 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 6
			if zerot && n < imat-2 {
				goto label160
			}

			//           Do first for uplo='U', then for uplo='L'
			for _, uplo = range mat.IterMatUplo(false) {
				if uplo == Upper {
					packit = 'C'
				} else {
					packit = 'R'
				}

				if imat != ntypes {
					//                 Set up parameters with ZLATB4 and generate a test
					//                 matrix with Zlatms.
					_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

					*srnamt = "Zlatms"
					if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, packit, a.CMatrix(lda, opts), work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						goto label150
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
							if uplo == Upper {
								//                          Set the first IZERO rows and columns to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i2 = min(j, izero)
									for i = 1; i <= i2; i++ {
										a.SetRe(ioff+i-1, zero)
									}
									ioff = ioff + j
								}
							} else {
								//                          Set the last IZERO rows and columns to zero.
								ioff = 0
								for j = 1; j <= n; j++ {
									i1 = max(j, izero)
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
				} else {
					//                 Use a special block diagonal matrix to test alternate
					//                 code for the 2 x 2 blocks.
					zlatsp(uplo, n, a, &iseed)
				}

				//              Compute the L*D*L' or U*D*U' factorization of the matrix.
				npp = n * (n + 1) / 2
				goblas.Zcopy(npp, a.Off(0, 1), afac.Off(0, 1))
				*srnamt = "Zsptrf"
				info, err = golapack.Zsptrf(uplo, n, afac, &iwork)

				//              Adjust the expected value of INFO to account for
				//              pivoting.
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

				//              Check error code from Zsptrf.
				if err != nil || info != k {
					t.Fail()
					nerrs = alaerh(path, "Zsptrf", info, k, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}
				if info != 0 {
					trfcon = true
				} else {
					trfcon = false
				}

				//+    TEST 1
				//              Reconstruct matrix from factors and compute residual.
				*result.GetPtr(0) = zspt01(uplo, n, a, afac, &iwork, ainv.CMatrix(lda, opts), rwork)
				nt = 1

				//+    TEST 2
				//              Form the inverse and compute the residual.
				if !trfcon {
					goblas.Zcopy(npp, afac.Off(0, 1), ainv.Off(0, 1))
					*srnamt = "Zsptri"
					if info, err = golapack.Zsptri(uplo, n, ainv, &iwork, work); err != nil || info != 0 {
						t.Fail()
						nerrs = alaerh(path, "Zsptri", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					rcondc, *result.GetPtr(1) = zspt03(uplo, n, a, ainv, work.CMatrix(lda, opts), rwork)
					nt = 2
				}

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = 1; k <= nt; k++ {
					if result.Get(k-1) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, k, result.Get(k-1))
						nfail++
					}
				}
				nrun = nrun + nt

				//              Do only the condition estimate if INFO is not 0.
				if trfcon {
					rcondc = zero
					goto label140
				}

				for irhs = 1; irhs <= nns; irhs++ {
					nrhs = nsval[irhs-1]

					//+    TEST 3
					//              Solve and compute residual for  A * X = B.
					*srnamt = "zlarhs"
					if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
						panic(err)
					}
					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

					*srnamt = "Zsptrs"
					if err = golapack.Zsptrs(uplo, n, nrhs, afac, &iwork, x.CMatrix(lda, opts)); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zsptrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
					*result.GetPtr(2) = zspt02(uplo, n, nrhs, a, x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

					//+    TEST 4
					//              Check solution from generated exact solution.
					*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

					//+    TESTS 5, 6, and 7
					//              Use iterative refinement to improve the solution.
					*srnamt = "Zsprfs"
					if err = golapack.Zsprfs(uplo, n, nrhs, a, afac, &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zsprfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					*result.GetPtr(4) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
					zppt05(uplo, n, nrhs, a, b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(5))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 3; k <= 7; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo='%c', n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 5
				}

				//+    TEST 8
				//              Get an estimate of RCOND = 1/CNDNUM.
			label140:
				;
				anorm = golapack.Zlansp('1', uplo, n, a, rwork)
				*srnamt = "Zspcon"
				if rcond, err = golapack.Zspcon(uplo, n, afac, &iwork, anorm, work); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zspcon", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				result.Set(7, dget06(rcond, rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(7) >= thresh {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					fmt.Printf(" uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, 8, result.Get(7))
					nfail++
				}
				nrun++
			label150:
			}
		label160:
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
