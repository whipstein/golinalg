package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dchktb tests DTBTRS, -RFS, and -CON, and Dlatbs.
func dchktb(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, ab, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var norm, xtype byte
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var diag mat.MatDiag
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, ik, imat, in, info, irhs, j, k, kd, lda, ldab, n, nerrs, nfail, nimat, nimat2, nk, nrhs, nrun, ntype1, ntypes int
	var err error

	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	ntype1 = 9
	ntypes = 17
	one = 1.0
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dtb"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrtr(path, t)
	}
	(*infot) = 0

	for in = 1; in <= nn; in++ {
		//        Do for each value of N in NVAL
		n = nval[in-1]
		lda = max(1, n)
		xtype = 'N'
		nimat = ntype1
		nimat2 = ntypes
		if n <= 0 {
			nimat = 1
			nimat2 = ntype1 + 1
		}

		nk = min(n+1, int(4))
		for ik = 1; ik <= nk; ik++ {
			//           Do for kd = 0, N, (3N-1)/4, and (N+1)/4. This order makes
			//           it easier to skip redundant values for small values of N.
			if ik == 1 {
				kd = 0
			} else if ik == 2 {
				kd = max(n, 0)
			} else if ik == 3 {
				kd = (3*n - 1) / 4
			} else if ik == 4 {
				kd = (n + 1) / 4
			}
			ldab = kd + 1

			for imat = 1; imat <= nimat; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !dotype[imat-1] {
					goto label90
				}

				for _, uplo = range mat.IterMatUplo(false) {
					//                 Do first for uplo = 'U', then for uplo = 'L'

					//                 Call DLATTB to generate a triangular test matrix.
					*srnamt = "Dlattb"
					diag, err = dlattb(imat, uplo, NoTrans, &iseed, n, kd, ab.Matrix(ldab, opts), x, work)

					//                 Form the inverse of A so we can get a good estimate
					//                 of RCONDC = 1/(norm(A) * norm(inv(A))).
					golapack.Dlaset(Full, n, n, zero, one, ainv.Matrix(lda, opts))
					if uplo == Upper {
						for j = 1; j <= n; j++ {
							err = ainv.Off((j-1)*lda).Tbsv(uplo, NoTrans, diag, j, kd, ab.Matrix(ldab, opts), 1)
						}
					} else {
						for j = 1; j <= n; j++ {
							err = ainv.Off((j-1)*lda+j-1).Tbsv(uplo, NoTrans, diag, n-j+1, kd, ab.Off((j-1)*ldab).Matrix(ldab, opts), 1)
						}
					}

					//                 Compute the 1-norm condition number of A.
					anorm = golapack.Dlantb('1', uplo, diag, n, kd, ab.Matrix(ldab, opts), rwork)
					ainvnm = golapack.Dlantr('1', uplo, diag, n, n, ainv.Matrix(lda, opts), rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondo = one
					} else {
						rcondo = (one / anorm) / ainvnm
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Dlantb('I', uplo, diag, n, kd, ab.Matrix(ldab, opts), rwork)
					ainvnm = golapack.Dlantr('I', uplo, diag, n, n, ainv.Matrix(lda, opts), rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]
						xtype = 'N'

						for _, trans = range mat.IterMatTrans() {
							//                    Do for op(A) = A, A**T, or A**H.
							if trans == NoTrans {
								norm = 'O'
								rcondc = rcondo
							} else {
								norm = 'I'
								rcondc = rcondi
							}

							//+    TEST 1
							//                    Solve and compute residual for op(A)*x = b.
							*srnamt = "Dlarhs"
							if err = Dlarhs(path, xtype, uplo, trans, n, n, kd, int(diag)+1, nrhs, ab.Matrix(ldab, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

							*srnamt = "Dtbtrs"
							if info, err = golapack.Dtbtrs(uplo, trans, diag, n, kd, nrhs, ab.Matrix(ldab, opts), x.Matrix(lda, opts)); err != nil || info != 0 {
								nerrs = alaerh(path, "Dtbtrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							result.Set(0, dtbt02(uplo, trans, diag, n, kd, nrhs, ab.Matrix(ldab, opts), x.Matrix(lda, opts), b.Matrix(lda, opts), work))

							//+    TEST 2
							//                    Check solution from generated exact solution.
							result.Set(1, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

							//+    TESTS 3, 4, and 5
							//                    Use iterative refinement to improve the solution
							//                    and compute error bounds.
							*srnamt = "Dtbrfs"
							if err = golapack.Dtbrfs(uplo, trans, diag, n, kd, nrhs, ab.Matrix(ldab, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
								nerrs = alaerh(path, "Dtbrfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							dtbt05(uplo, trans, diag, n, kd, nrhs, ab.Matrix(ldab, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= 5; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									t.Fail()
									fmt.Printf(" uplo=%s, trans=%s,        diag=%s, n=%5d, kd=%5d, nrhs=%5d, type %2d, test(%2d)=%12.5f\n", uplo, trans, diag, n, kd, nrhs, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 5
						}
					}

					//+    TEST 6
					//                    Get an estimate of RCOND = 1/CNDNUM.
					for _, trans = range mat.IterMatTrans(false) {
						if trans == NoTrans {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}
						*srnamt = "Dtbcon"
						if rcond, err = golapack.Dtbcon(norm, uplo, diag, n, kd, ab.Matrix(ldab, opts), work, &iwork); err != nil {
							nerrs = alaerh(path, "Dtbcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						result.Set(5, dtbt06(rcond, rcondc, uplo, diag, n, kd, ab.Matrix(ldab, opts), rwork))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(5) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" %s( '%c', %s, %s,%5d,%5d,  ... ), type %2d, test(%2d)=%12.5f\n", []byte("dtbcon"), norm, uplo, diag, n, kd, imat, 6, result.Get(5))
							nfail++
						}
						nrun++
					}
				}
			label90:
			}

			//           Use pathological test matrices to test Dlatbs.
			for imat = ntype1 + 1; imat <= nimat2; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !dotype[imat-1] {
					goto label120
				}

				for _, uplo = range mat.IterMatUplo(false) {
					//                 Do first for uplo = 'U', then for uplo = 'L'
					for _, trans = range mat.IterMatTrans() {
						//                    Do for op(A) = A, A**T, and A**H.

						//                    Call DLATTB to generate a triangular test matrix.
						*srnamt = "Dlattb"
						diag, err = dlattb(imat, uplo, trans, &iseed, n, kd, ab.Matrix(ldab, opts), x, work)

						//+    TEST 7
						//                    Solve the system op(A)*x = b
						*srnamt = "Dlatbs"
						b.Copy(n, x, 1, 1)
						if scale, err = golapack.Dlatbs(uplo, trans, diag, 'N', n, kd, ab.Matrix(ldab, opts), b, rwork); err != nil {
							nerrs = alaerh(path, "Dlatbs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						result.Set(6, dtbt03(uplo, trans, diag, n, kd, 1, ab.Matrix(ldab, opts), scale, rwork, one, b.Matrix(lda, opts), x.Matrix(lda, opts), work))

						//+    TEST 8
						//                    Solve op(A)*x = b again with NORMIN = 'Y'.
						b.Copy(n, x, 1, 1)
						if scale, err = golapack.Dlatbs(uplo, trans, diag, 'Y', n, kd, ab.Matrix(ldab, opts), b, rwork); err != nil {
							nerrs = alaerh(path, "Dlatbs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						result.Set(7, dtbt03(uplo, trans, diag, n, kd, 1, ab.Matrix(ldab, opts), scale, rwork, one, b.Matrix(lda, opts), x.Matrix(lda, opts), work))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(6) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" %s( %s, %s, %s, '%c',%5d,%5d, ...  ),  type %2d, test(%1d)=%12.5f\n", []byte("Dlatbs"), uplo, trans, diag, 'N', n, kd, imat, 7, result.Get(6))
							nfail++
						}
						if result.Get(7) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" %s( %s, %s, %s, '%c',%5d,%5d, ...  ),  type %2d, test(%1d)=%12.5f\n", []byte("Dlatbs"), uplo, trans, diag, 'Y', n, kd, imat, 8, result.Get(7))
							nfail++
						}
						nrun += 2
					}
				}
			label120:
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 19888
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
