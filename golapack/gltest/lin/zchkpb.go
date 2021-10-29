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

// zchkpb tests Zpbtrf, -TRS, -RFS, and -CON.
func zchkpb(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var zerot bool
	var dist, packit, _type, xtype byte
	var uplo mat.MatUplo
	var ainvnm, anorm, cndnum, one, rcond, rcondc, zero float64
	var i, i1, i2, ikd, imat, in, inb, info, ioff, irhs, iw, izero, k, kd, koff, lda, ldab, mode, n, nb, nerrs, nfail, nimat, nkd, nrhs, nrun, ntypes int
	var err error

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
	path := "Zpb"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrpo(path, t)
	}
	(*infot) = 0
	kdval[0] = 0

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
			//           Do for kd = 0, (5*N+1)/4, (3N-1)/4, and (N+1)/4. This order
			//           makes it easier to skip redundant values for small values
			//           of N.
			kd = kdval[ikd-1]
			ldab = kd + 1

			//           Do first for uplo = 'U', then for uplo = 'L'
			for _, uplo = range mat.IterMatUplo(false) {
				koff = 1
				if uplo == Upper {
					koff = max(1, kd+2-n)
					packit = 'Q'
				} else {
					packit = 'B'
				}

				for imat = 1; imat <= nimat; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !dotype[imat-1] {
						goto label60
					}

					//                 Skip types 2, 3, or 4 if the matrix size is too small.
					zerot = imat >= 2 && imat <= 4
					if zerot && n < imat-1 {
						goto label60
					}

					if !zerot || !dotype[0] {
						//                    Set up parameters with ZLATB4 and generate a test
						//                    matrix with Zlatms.
						_type, _, _, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

						*srnamt = "Zlatms"
						if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kd, kd, packit, a.CMatrixOff(koff-1, ldab, opts), work); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
							goto label60
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

					//                 Do for each value of NB in NBVAL
					for inb = 1; inb <= nnb; inb++ {
						nb = nbval[inb-1]
						xlaenv(1, nb)

						//                    Compute the L*L' or U'*U factorization of the band
						//                    matrix.
						golapack.Zlacpy(Full, kd+1, n, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts))
						*srnamt = "Zpbtrf"
						if info, err = golapack.Zpbtrf(uplo, n, kd, afac.CMatrix(ldab, opts)); err != nil || info != izero {
							t.Fail()
							nerrs = alaerh(path, "Zpbtrf", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nb, imat, nfail, nerrs)
							goto label50
						}

						//                    Skip the tests if INFO is not 0.
						if info != 0 {
							goto label50
						}

						//+    TEST 1
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						golapack.Zlacpy(Full, kd+1, n, afac.CMatrix(ldab, opts), ainv.CMatrix(ldab, opts))
						*result.GetPtr(0) = zpbt01(uplo, n, kd, a.CMatrix(ldab, opts), ainv.CMatrix(ldab, opts), rwork)

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(0) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, kd=%5d, NB=%4d, _type %2d, test %2d, ratio= %12.5f\n", uplo, n, kd, nb, imat, 1, result.Get(0))
							nfail++
						}
						nrun++

						//                    Only do other tests if this is the first blocksize.
						if inb > 1 {
							goto label50
						}

						//                    Form the inverse of A so we can get a good estimate
						//                    of RCONDC = 1/(norm(A) * norm(inv(A))).
						golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), ainv.CMatrix(lda, opts))
						*srnamt = "Zpbtrs"
						if err = golapack.Zpbtrs(uplo, n, kd, n, afac.CMatrix(ldab, opts), ainv.CMatrix(lda, opts)); err != nil {
							panic(err)
						}

						//                    Compute RCONDC = 1/(norm(A) * norm(inv(A))).
						anorm = golapack.Zlanhb('1', uplo, n, kd, a.CMatrix(ldab, opts), rwork)
						ainvnm = golapack.Zlange('1', n, n, ainv.CMatrix(lda, opts), rwork)
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}

						for irhs = 1; irhs <= nns; irhs++ {
							nrhs = nsval[irhs-1]

							//+    TEST 2
							//                    Solve and compute residual for A * X = B.
							*srnamt = "zlarhs"
							if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kd, kd, nrhs, a.CMatrix(ldab, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

							*srnamt = "Zpbtrs"
							if err = golapack.Zpbtrs(uplo, n, kd, nrhs, afac.CMatrix(ldab, opts), x.CMatrix(lda, opts)); err != nil {
								panic(err)
							}

							//                    Check error code from Zpbtrs.
							if info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Zpbtrs", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
							*result.GetPtr(1) = zpbt02(uplo, n, kd, nrhs, a.CMatrix(ldab, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

							//+    TEST 3
							//                    Check solution from generated exact solution.
							*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

							//+    TESTS 4, 5, and 6
							//                    Use iterative refinement to improve the solution.
							*srnamt = "Zpbrfs"
							if err = golapack.Zpbrfs(uplo, n, kd, nrhs, a.CMatrix(ldab, opts), afac.CMatrix(ldab, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
								panic(err)
							}

							//                    Check error code from Zpbrfs.
							if info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Zpbrfs", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							zpbt05(uplo, n, kd, nrhs, a.CMatrix(ldab, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" uplo=%s, n=%5d, kd=%5d, nrhs=%3d, _type %2d, test(%2d) = %12.5f\n", uplo, n, kd, nrhs, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 5
						}

						//+    TEST 7
						//                    Get an estimate of RCOND = 1/CNDNUM.
						*srnamt = "Zpbcon"
						if rcond, err = golapack.Zpbcon(uplo, n, kd, afac.CMatrix(ldab, opts), anorm, work, rwork); err != nil {
							panic(err)
						}

						//                    Check error code from Zpbcon.
						if info != 0 {
							nerrs = alaerh(path, "Zpbcon", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						result.Set(6, dget06(rcond, rcondc))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, kd=%5d,           _type %2d, test(%2d) = %12.5f\n", uplo, n, kd, imat, 7, result.Get(6))
							nfail++
						}
						nrun++
					label50:
					}
				label60:
				}
			}
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
