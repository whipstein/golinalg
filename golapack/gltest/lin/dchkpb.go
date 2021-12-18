package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkpb tests DPBTRF, -TRS, -RFS, and -CON.
func dchkpb(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
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
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 8

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dpb"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrpo(path, t)
	}
	(*infot) = 0
	xlaenv(2, 2)
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
			//
			//           Do for kd = 0, (5*N+1)/4, (3N-1)/4, and (N+1)/4. This order
			//           makes it easier to skip redundant values for small values
			//           of N.
			//
			kd = kdval[ikd-1]
			ldab = kd + 1
			//
			//           Do first for uplo = 'U', then for uplo = 'L'
			//
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
						//                    Set up parameters with DLATB4 and generate a test
						//                    matrix with DLATMS.
						_type, _, _, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)

						*srnamt = "Dlatms"
						if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kd, kd, packit, a.Off(koff-1).Matrix(ldab, opts), work); info != 0 {
							nerrs = alaerh(path, "Dlatms", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
							goto label60
						}
					} else if izero > 0 {
						//                    Use the same matrix for types 3 and 4 as for _type
						//                    2 by copying back the zeroed out column,
						iw = 2*lda + 1
						if uplo == Upper {
							ioff = (izero-1)*ldab + kd + 1
							a.Off(ioff-izero+i1-1).Copy(izero-i1, work.Off(iw-1), 1, 1)
							iw = iw + izero - i1
							a.Off(ioff-1).Copy(i2-izero+1, work.Off(iw-1), 1, max(ldab-1, 1))
						} else {
							ioff = (i1-1)*ldab + 1
							a.Off(ioff+izero-i1-1).Copy(izero-i1, work.Off(iw-1), 1, max(ldab-1, 1))
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							a.Off(ioff-1).Copy(i2-izero+1, work.Off(iw-1), 1, 1)
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
							work.Set(iw+i-1, zero)
						}
						iw = iw + 1
						i1 = max(izero-kd, 1)
						i2 = min(izero+kd, n)

						if uplo == Upper {
							ioff = (izero-1)*ldab + kd + 1
							work.Off(iw-1).Swap(izero-i1, a.Off(ioff-izero+i1-1), 1, 1)
							iw = iw + izero - i1
							work.Off(iw-1).Swap(i2-izero+1, a.Off(ioff-1), max(ldab-1, 1), 1)
						} else {
							ioff = (i1-1)*ldab + 1
							work.Off(iw-1).Swap(izero-i1, a.Off(ioff+izero-i1-1), max(ldab-1, 1), 1)
							ioff = (izero-1)*ldab + 1
							iw = iw + izero - i1
							work.Off(iw-1).Swap(i2-izero+1, a.Off(ioff-1), 1, 1)
						}
					}

					//                 Do for each value of NB in NBVAL
					for inb = 1; inb <= nnb; inb++ {
						nb = nbval[inb-1]
						xlaenv(1, nb)

						//                    Compute the L*L' or U'*U factorization of the band
						//                    matrix.
						golapack.Dlacpy(Full, kd+1, n, a.Matrix(ldab, opts), afac.Matrix(ldab, opts))
						*srnamt = "Dpbtrf"
						if info, err = golapack.Dpbtrf(uplo, n, kd, afac.Matrix(ldab, opts)); err != nil {
							panic(err)
						}

						//                    Check error code from DPBTRF.
						if info != izero {
							nerrs = alaerh(path, "Dpbtrf", info, izero, []byte{uplo.Byte()}, n, n, kd, kd, nb, imat, nfail, nerrs)
							goto label50
						}

						//                    Skip the tests if INFO is not 0.
						if info != 0 {
							goto label50
						}

						//+    TEST 1
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						golapack.Dlacpy(Full, kd+1, n, afac.Matrix(ldab, opts), ainv.Matrix(ldab, opts))
						result.Set(0, dpbt01(uplo, n, kd, a.Matrix(ldab, opts), ainv.Matrix(ldab, opts), rwork))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(0) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
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
						golapack.Dlaset(Full, n, n, zero, one, ainv.Matrix(lda, opts))
						*srnamt = "Dpbtrs"
						if err = golapack.Dpbtrs(uplo, n, kd, n, afac.Matrix(ldab, opts), ainv.Matrix(lda, opts)); err != nil {
							panic(err)
						}

						//                    Compute RCONDC = 1/(norm(A) * norm(inv(A))).
						anorm = golapack.Dlansb('1', uplo, n, kd, a.Matrix(ldab, opts), rwork)
						ainvnm = golapack.Dlange('1', n, n, ainv.Matrix(lda, opts), rwork)
						if anorm <= zero || ainvnm <= zero {
							rcondc = one
						} else {
							rcondc = (one / anorm) / ainvnm
						}

						for irhs = 1; irhs <= nns; irhs++ {
							nrhs = nsval[irhs-1]

							//+    TEST 2
							//                    Solve and compute residual for A * X = B.
							*srnamt = "Dlarhs"
							if err = Dlarhs(path, xtype, uplo, NoTrans, n, n, kd, kd, nrhs, a.Matrix(ldab, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

							*srnamt = "Dpbtrs"
							if err = golapack.Dpbtrs(uplo, n, kd, nrhs, afac.Matrix(ldab, opts), x.Matrix(lda, opts)); err != nil {
								panic(err)
							}

							//                    Check error code from DPBTRS.
							if info != 0 {
								nerrs = alaerh(path, "Dpbtrs", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
							result.Set(1, dpbt02(uplo, n, kd, nrhs, a.Matrix(ldab, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

							//+    TEST 3
							//                    Check solution from generated exact solution.
							result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

							//+    TESTS 4, 5, and 6
							//                    Use iterative refinement to improve the solution.
							*srnamt = "Dpbrfs"
							if err = golapack.Dpbrfs(uplo, n, kd, nrhs, a.Matrix(ldab, opts), afac.Matrix(ldab, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
								nerrs = alaerh(path, "Dpbrfs", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							dpbt05(uplo, n, kd, nrhs, a.Matrix(ldab, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
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
						*srnamt = "Dpbcon"
						if rcond, err = golapack.Dpbcon(uplo, n, kd, afac.Matrix(ldab, opts), anorm, work, &iwork); err != nil {
							nerrs = alaerh(path, "Dpbcon", info, 0, []byte{uplo.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						result.Set(6, dget06(rcond, rcondc))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
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

	//     Verify number of tests match original.
	tgtRuns := 3458
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
