package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zchktb tests ZTBTRS, -RFS, and -CON, and ZLATBS.
func Zchktb(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, ab, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var diag, norm, trans, uplo, xtype byte
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, ik, imat, in, info, irhs, itran, iuplo, j, k, kd, lda, ldab, n, nerrs, nfail, nimat, nimat2, nk, nrhs, nrun, ntran, ntype1, ntypes int
	var err error
	_ = err

	transs := make([]byte, 3)
	uplos := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntype1 = 9
	ntypes = 17
	ntran = 3
	one = 1.0
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], transs[0], transs[1], transs[2] = 'U', 'L', 'N', 'T', 'C'

	//     Initialize constants and the random number seed.
	path := []byte("ZTB")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrtr(path, t)
	}
	(*infot) = 0

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL
		n = (*nval)[in-1]
		lda = maxint(1, n)
		xtype = 'N'
		nimat = ntype1
		nimat2 = ntypes
		if n <= 0 {
			nimat = 1
			nimat2 = ntype1 + 1
		}

		nk = minint(n+1, int(4))
		for ik = 1; ik <= nk; ik++ {
			//           Do for KD = 0, N, (3N-1)/4, and (N+1)/4. This order makes
			//           it easier to skip redundant values for small values of N.
			if ik == 1 {
				kd = 0
			} else if ik == 2 {
				kd = maxint(n, 0)
			} else if ik == 3 {
				kd = (3*n - 1) / 4
			} else if ik == 4 {
				kd = (n + 1) / 4
			}
			ldab = kd + 1

			for imat = 1; imat <= nimat; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !(*dotype)[imat-1] {
					goto label90
				}

				for iuplo = 1; iuplo <= 2; iuplo++ {
					//                 Do first for UPLO = 'U', then for UPLO = 'L'
					uplo = uplos[iuplo-1]

					//                 Call ZLATTB to generate a triangular test matrix.
					*srnamt = "ZLATTB"
					Zlattb(&imat, uplo, 'N', &diag, &iseed, &n, &kd, ab.CMatrix(ldab, opts), &ldab, x, work, rwork, &info)
					//
					//                 Set IDIAG = 1 for non-unit matrices, 2 for unit.
					//
					if diag == 'N' {
						idiag = 1
					} else {
						idiag = 2
					}
					//
					//                 Form the inverse of A so we can get a good estimate
					//                 of RCONDC = 1/(norm(A) * norm(inv(A))).
					//
					golapack.Zlaset('F', &n, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), ainv.CMatrix(lda, opts), &lda)
					if uplo == 'U' {
						for j = 1; j <= n; j++ {
							err = goblas.Ztbsv(mat.UploByte(uplo), NoTrans, mat.DiagByte(diag), j, kd, ab.CMatrix(ldab, opts), ldab, ainv.Off((j-1)*lda+1-1), 1)
						}
					} else {
						for j = 1; j <= n; j++ {
							err = goblas.Ztbsv(mat.UploByte(uplo), NoTrans, mat.DiagByte(diag), n-j+1, kd, ab.CMatrixOff((j-1)*ldab+1-1, ldab, opts), ldab, ainv.Off((j-1)*lda+j-1), 1)
						}
					}

					//                 Compute the 1-norm condition number of A.
					anorm = golapack.Zlantb('1', uplo, diag, &n, &kd, ab.CMatrix(ldab, opts), &ldab, rwork)
					ainvnm = golapack.Zlantr('1', uplo, diag, &n, &n, ainv.CMatrix(lda, opts), &lda, rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondo = one
					} else {
						rcondo = (one / anorm) / ainvnm
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Zlantb('I', uplo, diag, &n, &kd, ab.CMatrix(ldab, opts), &ldab, rwork)
					ainvnm = golapack.Zlantr('I', uplo, diag, &n, &n, ainv.CMatrix(lda, opts), &lda, rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]
						xtype = 'N'

						for itran = 1; itran <= ntran; itran++ {
							//                    Do for op(A) = A, A**T, or A**H.
							trans = transs[itran-1]
							if itran == 1 {
								norm = 'O'
								rcondc = rcondo
							} else {
								norm = 'I'
								rcondc = rcondi
							}

							//+    TEST 1
							//                    Solve and compute residual for op(A)*x = b.
							*srnamt = "ZLARHS"
							Zlarhs(path, xtype, uplo, trans, &n, &n, &kd, &idiag, &nrhs, ab.CMatrix(ldab, opts), &ldab, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
							xtype = 'C'
							golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

							*srnamt = "ZTBTRS"
							golapack.Ztbtrs(uplo, trans, diag, &n, &kd, &nrhs, ab.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, &info)

							//                    Check error code from ZTBTRS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZTBTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, &kd, &kd, &nrhs, &imat, &nfail, &nerrs)
							}

							Ztbt02(uplo, trans, diag, &n, &kd, &nrhs, ab.CMatrix(ldab, opts), &ldab, x.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, work, rwork, result.GetPtr(0))

							//+    TEST 2
							//                    Check solution from generated exact solution.
							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(1))

							//+    TESTS 3, 4, and 5
							//                    Use iterative refinement to improve the solution
							//                    and compute error bounds.
							*srnamt = "ZTBRFS"
							golapack.Ztbrfs(uplo, trans, diag, &n, &kd, &nrhs, ab.CMatrix(ldab, opts), &ldab, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

							//                    Check error code from ZTBRFS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZTBRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, &kd, &kd, &nrhs, &imat, &nfail, &nerrs)
							}

							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							Ztbt05(uplo, trans, diag, &n, &kd, &nrhs, ab.CMatrix(ldab, opts), &ldab, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(3))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 1; k <= 5; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" UPLO='%c', TRANS='%c',        DIAG='%c', N=%5d, KD=%5d, NRHS=%5d, _type %2d, test(%2d)=%12.5f\n", uplo, trans, diag, n, kd, nrhs, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 5
						}
					}

					//+    TEST 6
					//                    Get an estimate of RCOND = 1/CNDNUM.
					for itran = 1; itran <= 2; itran++ {
						if itran == 1 {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}
						*srnamt = "ZTBCON"
						golapack.Ztbcon(norm, uplo, diag, &n, &kd, ab.CMatrix(ldab, opts), &ldab, &rcond, work, rwork, &info)

						//                    Check error code from ZTBCON.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZTBCON"), &info, func() *int { y := 0; return &y }(), []byte{norm, uplo, diag}, &n, &n, &kd, &kd, toPtr(-1), &imat, &nfail, &nerrs)
						}

						Ztbt06(&rcond, &rcondc, uplo, diag, &n, &kd, ab.CMatrix(ldab, opts), &ldab, rwork, result.GetPtr(5))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(5) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" %s( '%c', '%c', '%c',%5d,%5d,  ... ), _type %2d, test(%2d)=%12.5f\n", "ZTBCON", norm, uplo, diag, n, kd, imat, 6, result.Get(5))
							nfail = nfail + 1
						}
						nrun = nrun + 1
					}
				}
			label90:
			}

			//           Use pathological test matrices to test ZLATBS.
			for imat = ntype1 + 1; imat <= nimat2; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !(*dotype)[imat-1] {
					goto label120
				}
				for iuplo = 1; iuplo <= 2; iuplo++ {
					//                 Do first for UPLO = 'U', then for UPLO = 'L'
					uplo = uplos[iuplo-1]
					for itran = 1; itran <= ntran; itran++ {
						//                    Do for op(A) = A, A**T, and A**H.
						trans = transs[itran-1]

						//                    Call ZLATTB to generate a triangular test matrix.
						*srnamt = "ZLATTB"
						Zlattb(&imat, uplo, trans, &diag, &iseed, &n, &kd, ab.CMatrix(ldab, opts), &ldab, x, work, rwork, &info)

						//+    TEST 7
						//                    Solve the system op(A)*x = b
						*srnamt = "ZLATBS"
						goblas.Zcopy(n, x, 1, b, 1)
						golapack.Zlatbs(uplo, trans, diag, 'N', &n, &kd, ab.CMatrix(ldab, opts), &ldab, b, &scale, rwork, &info)

						//                    Check error code from ZLATBS.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZLATBS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'N'}, &n, &n, &kd, &kd, toPtr(-1), &imat, &nfail, &nerrs)
						}

						Ztbt03(uplo, trans, diag, &n, &kd, func() *int { y := 1; return &y }(), ab.CMatrix(ldab, opts), &ldab, &scale, rwork, &one, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work, result.GetPtr(6))

						//+    TEST 8
						//                    Solve op(A)*x = b again with NORMIN = 'Y'.
						goblas.Zcopy(n, x, 1, b, 1)
						golapack.Zlatbs(uplo, trans, diag, 'Y', &n, &kd, ab.CMatrix(ldab, opts), &ldab, b, &scale, rwork, &info)

						//                    Check error code from ZLATBS.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZLATBS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'Y'}, &n, &n, &kd, &kd, toPtr(-1), &imat, &nfail, &nerrs)
						}

						Ztbt03(uplo, trans, diag, &n, &kd, func() *int { y := 1; return &y }(), ab.CMatrix(ldab, opts), &ldab, &scale, rwork, &one, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work, result.GetPtr(7))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(6) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d,%5d, ...  ),  _type %2d, test(%1d)=%12.5f\n", "ZLATBS", uplo, trans, diag, 'N', n, kd, imat, 7, result.Get(6))
							nfail = nfail + 1
						}
						if result.Get(7) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d,%5d, ...  ),  _type %2d, test(%1d)=%12.5f\n", "ZLATBS", uplo, trans, diag, 'Y', n, kd, imat, 8, result.Get(7))
							nfail = nfail + 1
						}
						nrun = nrun + 2
					}
				}
			label120:
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
