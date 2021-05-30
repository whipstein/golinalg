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

// Zchkpb tests ZPBTRF, -TRS, -RFS, and -CON.
func Zchkpb(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var zerot bool
	var dist, packit, _type, uplo, xtype byte
	var ainvnm, anorm, cndnum, one, rcond, rcondc, zero float64
	var i, i1, i2, ikd, imat, in, inb, info, ioff, irhs, iuplo, iw, izero, k, kd, kl, koff, ku, lda, ldab, mode, n, nb, nerrs, nfail, nimat, nkd, nrhs, nrun, ntypes int

	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kdval := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 8
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

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
		Zerrpo(path, t)
	}
	(*infot) = 0
	kdval[0] = 0

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
		xtype = 'N'

		//        Set limits on the number of loop iterations.
		nkd = maxint(1, minint(n, 4))
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
					koff = maxint(1, kd+2-n)
					packit = 'Q'
				} else {
					uplo = 'L'
					packit = 'B'
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !(*dotype)[imat-1] {
						goto label60
					}

					//                 Skip types 2, 3, or 4 if the matrix size is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label60
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
							Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, &kd, &kd, toPtr(-1), &imat, &nfail, &nerrs)
							goto label60
						}
					} else if izero > 0 {
						//                    Use the same matrix for types 3 and 4 as for _type
						//                    2 by copying back the zeroed out column,
						iw = 2*lda + 1
						if iuplo == 1 {
							ioff = (izero-1)*ldab + kd + 1
							goblas.Zcopy(toPtr(izero-i1), work.Off(iw-1), func() *int { y := 1; return &y }(), a.Off(ioff-izero+i1-1), func() *int { y := 1; return &y }())
							iw = iw + izero - i1
							goblas.Zcopy(toPtr(i2-izero+1), work.Off(iw-1), func() *int { y := 1; return &y }(), a.Off(ioff-1), toPtr(maxint(ldab-1, 1)))
						} else {
							ioff = (i1-1)*ldab + 1
							goblas.Zcopy(toPtr(izero-i1), work.Off(iw-1), func() *int { y := 1; return &y }(), a.Off(ioff+izero-i1-1), toPtr(maxint(ldab-1, 1)))
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							goblas.Zcopy(toPtr(i2-izero+1), work.Off(iw-1), func() *int { y := 1; return &y }(), a.Off(ioff-1), func() *int { y := 1; return &y }())
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
						for i = 1; i <= minint(2*kd+1, n); i++ {
							work.SetRe(iw+i-1, zero)
						}
						iw = iw + 1
						i1 = maxint(izero-kd, 1)
						i2 = minint(izero+kd, n)

						if iuplo == 1 {
							ioff = (izero-1)*ldab + kd + 1
							goblas.Zswap(toPtr(izero-i1), a.Off(ioff-izero+i1-1), func() *int { y := 1; return &y }(), work.Off(iw-1), func() *int { y := 1; return &y }())
							iw = iw + izero - i1
							goblas.Zswap(toPtr(i2-izero+1), a.Off(ioff-1), toPtr(maxint(ldab-1, 1)), work.Off(iw-1), func() *int { y := 1; return &y }())
						} else {
							ioff = (i1-1)*ldab + 1
							goblas.Zswap(toPtr(izero-i1), a.Off(ioff+izero-i1-1), toPtr(maxint(ldab-1, 1)), work.Off(iw-1), func() *int { y := 1; return &y }())
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							goblas.Zswap(toPtr(i2-izero+1), a.Off(ioff-1), func() *int { y := 1; return &y }(), work.Off(iw-1), func() *int { y := 1; return &y }())
						}
					}

					//                 Set the imaginary part of the diagonals.
					if iuplo == 1 {
						Zlaipd(&n, a.Off(kd+1-1), &ldab, func() *int { y := 0; return &y }())
					} else {
						Zlaipd(&n, a.Off(0), &ldab, func() *int { y := 0; return &y }())
					}

					//                 Do for each value of NB in NBVAL
					for inb = 1; inb <= (*nnb); inb++ {
						nb = (*nbval)[inb-1]
						Xlaenv(1, nb)

						//                    Compute the L*L' or U'*U factorization of the band
						//                    matrix.
						golapack.Zlacpy('F', toPtr(kd+1), &n, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab)
						*srnamt = "ZPBTRF"
						golapack.Zpbtrf(uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, &info)

						//                    Check error code from ZPBTRF.
						if info != izero {
							t.Fail()
							Alaerh(path, []byte("ZPBTRF"), &info, &izero, []byte{uplo}, &n, &n, &kd, &kd, &nb, &imat, &nfail, &nerrs)
							goto label50
						}

						//                    Skip the tests if INFO is not 0.
						if info != 0 {
							goto label50
						}

						//+    TEST 1
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						golapack.Zlacpy('F', toPtr(kd+1), &n, afac.CMatrix(ldab, opts), &ldab, ainv.CMatrix(ldab, opts), &ldab)
						Zpbt01(uplo, &n, &kd, a.CMatrix(ldab, opts), &ldab, ainv.CMatrix(ldab, opts), &ldab, rwork, result.GetPtr(0))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(0) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO='%c', N=%5d, KD=%5d, NB=%4d, _type %2d, test %2d, ratio= %12.5f\n", uplo, n, kd, nb, imat, 1, result.Get(0))
							nfail = nfail + 1
						}
						nrun = nrun + 1

						//                    Only do other tests if this is the first blocksize.
						if inb > 1 {
							goto label50
						}

						//                    Form the inverse of A so we can get a good estimate
						//                    of RCONDC = 1/(norm(A) * norm(inv(A))).
						golapack.Zlaset('F', &n, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), ainv.CMatrix(lda, opts), &lda)
						*srnamt = "ZPBTRS"
						golapack.Zpbtrs(uplo, &n, &kd, &n, afac.CMatrix(ldab, opts), &ldab, ainv.CMatrix(lda, opts), &lda, &info)

						//                    Compute RCONDC = 1/(norm(A) * norm(inv(A))).
						anorm = golapack.Zlanhb('1', uplo, &n, &kd, a.CMatrix(ldab, opts), &ldab, rwork)
						ainvnm = golapack.Zlange('1', &n, &n, ainv.CMatrix(lda, opts), &lda, rwork)
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}

						for irhs = 1; irhs <= (*nns); irhs++ {
							nrhs = (*nsval)[irhs-1]

							//+    TEST 2
							//                    Solve and compute residual for A * X = B.
							*srnamt = "ZLARHS"
							Zlarhs(path, xtype, uplo, ' ', &n, &n, &kd, &kd, &nrhs, a.CMatrix(ldab, opts), &ldab, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
							golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

							*srnamt = "ZPBTRS"
							golapack.Zpbtrs(uplo, &n, &kd, &nrhs, afac.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, &info)

							//                    Check error code from ZPBTRS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZPBTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, &kd, &kd, &nrhs, &imat, &nfail, &nerrs)
							}

							golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
							Zpbt02(uplo, &n, &kd, &nrhs, a.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(1))

							//+    TEST 3
							//                    Check solution from generated exact solution.
							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

							//+    TESTS 4, 5, and 6
							//                    Use iterative refinement to improve the solution.
							*srnamt = "ZPBRFS"
							golapack.Zpbrfs(uplo, &n, &kd, &nrhs, a.CMatrix(ldab, opts), &ldab, afac.CMatrix(ldab, opts), &ldab, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

							//                    Check error code from ZPBRFS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZPBRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, &kd, &kd, &nrhs, &imat, &nfail, &nerrs)
							}

							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
							Zpbt05(uplo, &n, &kd, &nrhs, a.CMatrix(ldab, opts), &ldab, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" UPLO='%c', N=%5d, KD=%5d, NRHS=%3d, _type %2d, test(%2d) = %12.5f\n", uplo, n, kd, nrhs, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 5
						}

						//+    TEST 7
						//                    Get an estimate of RCOND = 1/CNDNUM.
						*srnamt = "ZPBCON"
						golapack.Zpbcon(uplo, &n, &kd, afac.CMatrix(ldab, opts), &ldab, &anorm, &rcond, work, rwork, &info)

						//                    Check error code from ZPBCON.
						if info != 0 {
							Alaerh(path, []byte("ZPBCON"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, &kd, &kd, toPtr(-1), &imat, &nfail, &nerrs)
						}

						result.Set(6, Dget06(&rcond, &rcondc))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO='%c', N=%5d, KD=%5d,           _type %2d, test(%2d) = %12.5f\n", uplo, n, kd, imat, 7, result.Get(6))
							nfail = nfail + 1
						}
						nrun = nrun + 1
					label50:
					}
				label60:
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
