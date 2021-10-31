package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zchktb tests Ztbtrs, -RFS, and -CON, and Zlatbs.
func zchktb(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, ab, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var norm, xtype byte
	var diag mat.MatDiag
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, ik, imat, in, info, irhs, j, k, kd, lda, ldab, n, nerrs, nfail, nimat, nimat2, nk, nrhs, nrun, ntype1, ntypes int
	var err error

	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntype1 = 9
	ntypes = 17
	one = 1.0
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Ztb"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrtr(path, t)
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
			//           Do for KD = 0, N, (3N-1)/4, and (N+1)/4. This order makes
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

					//                 Call Zlattb to generate a triangular test matrix.
					*srnamt = "Zlattb"
					diag = zlattb(imat, uplo, NoTrans, &iseed, n, kd, ab.CMatrix(ldab, opts), x, work, rwork)
					//
					//                 Set IDIAG = 1 for non-unit matrices, 2 for unit.
					//
					if diag == NonUnit {
						idiag = 1
					} else {
						idiag = 2
					}
					//
					//                 Form the inverse of A so we can get a good estimate
					//                 of RCONDC = 1/(norm(A) * norm(inv(A))).
					//
					golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), ainv.CMatrix(lda, opts))
					if uplo == Upper {
						for j = 1; j <= n; j++ {
							if err = goblas.Ztbsv(uplo, NoTrans, diag, j, kd, ab.CMatrix(ldab, opts), ainv.Off((j-1)*lda, 1)); err != nil {
								panic(err)
							}
						}
					} else {
						for j = 1; j <= n; j++ {
							if err = goblas.Ztbsv(uplo, NoTrans, diag, n-j+1, kd, ab.CMatrixOff((j-1)*ldab, ldab, opts), ainv.Off((j-1)*lda+j-1, 1)); err != nil {
								panic(err)
							}
						}
					}

					//                 Compute the 1-norm condition number of A.
					anorm = golapack.Zlantb('1', uplo, diag, n, kd, ab.CMatrix(ldab, opts), rwork)
					ainvnm = golapack.Zlantr('1', uplo, diag, n, n, ainv.CMatrix(lda, opts), rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondo = one
					} else {
						rcondo = (one / anorm) / ainvnm
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Zlantb('I', uplo, diag, n, kd, ab.CMatrix(ldab, opts), rwork)
					ainvnm = golapack.Zlantr('I', uplo, diag, n, n, ainv.CMatrix(lda, opts), rwork)
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
							*srnamt = "zlarhs"
							if err = zlarhs(path, xtype, uplo, trans, n, n, kd, idiag, nrhs, ab.CMatrix(ldab, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

							*srnamt = "Ztbtrs"
							if info, err = golapack.Ztbtrs(uplo, trans, diag, n, kd, nrhs, ab.CMatrix(ldab, opts), x.CMatrix(lda, opts)); err != nil || info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Ztbtrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							*result.GetPtr(0) = ztbt02(uplo, trans, diag, n, kd, nrhs, ab.CMatrix(ldab, opts), x.CMatrix(lda, opts), b.CMatrix(lda, opts), work, rwork)

							//+    TEST 2
							//                    Check solution from generated exact solution.
							*result.GetPtr(1) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

							//+    TESTS 3, 4, and 5
							//                    Use iterative refinement to improve the solution
							//                    and compute error bounds.
							*srnamt = "Ztbrfs"
							if err = golapack.Ztbrfs(uplo, trans, diag, n, kd, nrhs, ab.CMatrix(ldab, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
								t.Fail()
								nerrs = alaerh(path, "Ztbrfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, kd, kd, nrhs, imat, nfail, nerrs)
							}

							*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							ztbt05(uplo, trans, diag, n, kd, nrhs, ab.CMatrix(ldab, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= 5; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" uplo=%s, trans=%s,        diag=%s, n=%5d, kd=%5d, nrhs=%5d, _type %2d, test(%2d)=%12.5f\n", uplo, trans, diag, n, kd, nrhs, imat, k, result.Get(k-1))
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
						*srnamt = "Ztbcon"
						if rcond, err = golapack.Ztbcon(norm, uplo, diag, n, kd, ab.CMatrix(ldab, opts), work, rwork); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Ztbcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						*result.GetPtr(5) = ztbt06(rcond, rcondc, uplo, diag, n, kd, ab.CMatrix(ldab, opts), rwork)

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(5) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" %s( '%c', %s, %s,%5d,%5d,  ... ), _type %2d, test(%2d)=%12.5f\n", "Ztbcon", norm, uplo, diag, n, kd, imat, 6, result.Get(5))
							nfail++
						}
						nrun++
					}
				}
			label90:
			}

			//           Use pathological test matrices to test Zlatbs.
			for imat = ntype1 + 1; imat <= nimat2; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !dotype[imat-1] {
					goto label120
				}
				for _, uplo = range mat.IterMatUplo(false) {
					//                 Do first for uplo='U', then for uplo='L'
					for _, trans = range mat.IterMatTrans() {

						//                    Call Zlattb to generate a triangular test matrix.
						*srnamt = "Zlattb"
						diag = zlattb(imat, uplo, trans, &iseed, n, kd, ab.CMatrix(ldab, opts), x, work, rwork)

						//+    TEST 7
						//                    Solve the system op(A)*x = b
						*srnamt = "Zlatbs"
						goblas.Zcopy(n, x.Off(0, 1), b.Off(0, 1))
						if scale, err = golapack.Zlatbs(uplo, trans, diag, 'N', n, kd, ab.CMatrix(ldab, opts), b, rwork); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zlatbs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						*result.GetPtr(6) = ztbt03(uplo, trans, diag, n, kd, 1, ab.CMatrix(ldab, opts), scale, rwork, one, b.CMatrix(lda, opts), x.CMatrix(lda, opts), work)

						//+    TEST 8
						//                    Solve op(A)*x = b again with NORMIN = 'Y'.
						goblas.Zcopy(n, x.Off(0, 1), b.Off(0, 1))
						if scale, err = golapack.Zlatbs(uplo, trans, diag, 'Y', n, kd, ab.CMatrix(ldab, opts), b, rwork); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zlatbs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, kd, kd, -1, imat, nfail, nerrs)
						}

						*result.GetPtr(7) = ztbt03(uplo, trans, diag, n, kd, 1, ab.CMatrix(ldab, opts), scale, rwork, one, b.CMatrix(lda, opts), x.CMatrix(lda, opts), work)

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(6) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" %s( %s, %s, %s, '%c',%5d,%5d, ...  ),  _type %2d, test(%1d)=%12.5f\n", "Zlatbs", uplo, trans, diag, 'N', n, kd, imat, 7, result.Get(6))
							nfail++
						}
						if result.Get(7) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" %s( %s, %s, %s, '%c',%5d,%5d, ...  ),  _type %2d, test(%1d)=%12.5f\n", "Zlatbs", uplo, trans, diag, 'Y', n, kd, imat, 8, result.Get(7))
							nfail++
						}
						nrun += 2
					}
				}
			label120:
			}
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
