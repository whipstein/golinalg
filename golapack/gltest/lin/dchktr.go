package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dchktr tests DTRTRI, -TRS, -RFS, and -CON, and DLATRS
func dchktr(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var norm, xtype byte
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var diag mat.MatDiag
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var _result *float64
	var i, imat, in, inb, info, irhs, k, lda, n, nb, nerrs, nfail, nrhs, nrun, ntype1, ntypes int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	ntype1 = 10
	ntypes = 18
	one = 1.0
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dtr"
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
	xlaenv(2, 2)

	for in = 1; in <= nn; in++ {
		//        Do for each value of N in NVAL
		n = nval[in-1]
		lda = max(1, n)
		xtype = 'N'

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label80
			}

			for _, uplo = range mat.IterMatUplo(false) {
				//              Do first for uplo= 'U', then for uplo= 'L'

				//              Call DLATTR to generate a triangular test matrix.
				*srnamt = "Dlattr"
				diag, iseed, err = dlattr(imat, uplo, NoTrans, iseed, n, a.Matrix(lda, opts), x, work)

				for inb = 1; inb <= nnb; inb++ {
					//                 Do for each blocksize in NBVAL
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//+    TEST 1
					//                 Form the inverse of A.
					golapack.Dlacpy(uplo, n, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts))
					*srnamt = "Dtrtri"
					if info, err = golapack.Dtrtri(uplo, diag, n, ainv.Matrix(lda, opts)); err != nil {
						panic(err)
					}

					//                 Check error code from DTRTRI.
					if info != 0 {
						nerrs = alaerh(path, "Dtrtri", info, 0, []byte{uplo.Byte(), diag.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Dlantr('I', uplo, diag, n, n, a.Matrix(lda, opts), rwork)
					ainvnm = golapack.Dlantr('I', uplo, diag, n, n, ainv.Matrix(lda, opts), rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					//                 Compute the residual for the triangular matrix times
					//                 its inverse.  Also compute the 1-norm condition number
					//                 of A.
					_result = result.GetPtr(0)
					rcondo, *_result = dtrt01(uplo, diag, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), rwork)

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(0) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" uplo=%s, diag=%s, n=%5d, nb=%4d, type %2d, test(%2d)= %12.5f\n", uplo, diag, n, nb, imat, 1, result.Get(0))
						nfail++
					}
					nrun++

					//                 Skip remaining tests if not the first block size.
					if inb != 1 {
						goto label60
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

							//+    TEST 2
							//                       Solve and compute residual for op(A)*x = b.
							*srnamt = "Dlarhs"
							if err = Dlarhs(path, xtype, uplo, trans, n, n, 0, int(diag)+1, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

							*srnamt = "Dtrtrs"
							if info, err = golapack.Dtrtrs(uplo, trans, diag, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts)); err != nil || info != 0 {
								nerrs = alaerh(path, "Dtrtrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							result.Set(1, dtrt02(uplo, trans, diag, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), b.Matrix(lda, opts), work))

							//+    TEST 3
							//                       Check solution from generated exact solution.
							result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

							//+    TESTS 4, 5, and 6
							//                       Use iterative refinement to improve the solution
							//                       and compute error bounds.
							*srnamt = "Dtrrfs"
							if err = golapack.Dtrrfs(uplo, trans, diag, n, nrhs, a.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
								nerrs = alaerh(path, "Dtrrfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							dtrt05(uplo, trans, diag, n, nrhs, a.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									t.Fail()
									fmt.Printf(" uplo=%s, trans=%s, diag=%s, n=%5d, nb=%4d, type %2d,        test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 5
						}
					}

					//+    TEST 7
					//                       Get an estimate of RCOND = 1/CNDNUM.
					for _, trans = range mat.IterMatTrans(false) {
						if trans == NoTrans {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}
						*srnamt = "Dtrcon"
						if rcond, err = golapack.Dtrcon(norm, uplo, diag, n, a.Matrix(lda, opts), work, &iwork); err != nil {
							nerrs = alaerh(path, "Dtrcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						result.Set(6, dtrt06(rcond, rcondc, uplo, diag, n, a.Matrix(lda, opts), rwork))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" norm='%c', uplo=%s, n=%5d,            type %2d, test(%2d)=%12.5f\n", norm, uplo, n, imat, 7, result.Get(6))
							nfail++
						}
						nrun++
					}
				label60:
				}
			}
		label80:
		}

		//        Use pathological test matrices to test DLATRS.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label110
			}

			for _, uplo = range mat.IterMatUplo(false) {
				//              Do first for uplo= 'U', then for uplo= 'L'
				for _, trans = range mat.IterMatTrans() {
					//                 Do for op(A) = A, A**T, and A**H.

					//                 Call DLATTR to generate a triangular test matrix.
					*srnamt = "Dlattr"
					diag, iseed, err = dlattr(imat, uplo, trans, iseed, n, a.Matrix(lda, opts), x, work)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "Dlatrs"
					goblas.Dcopy(n, x.Off(0, 1), b.Off(0, 1))
					if scale, err = golapack.Dlatrs(uplo, trans, diag, 'N', n, a.Matrix(lda, opts), b, scale, rwork); err != nil {
						nerrs = alaerh(path, "Dlatrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(7, dtrt03(uplo, trans, diag, n, 1, a.Matrix(lda, opts), scale, rwork, one, b.Matrix(lda, opts), x.Matrix(lda, opts), work))

					//+    TEST 9
					//                 Solve op(A)*X = b again with NORMIN = 'Y'.
					goblas.Dcopy(n, x.Off(0, 1), b.Off(n, 1))
					if scale, err = golapack.Dlatrs(uplo, trans, diag, 'Y', n, a.Matrix(lda, opts), b.Off(n), scale, rwork); err != nil {
						nerrs = alaerh(path, "Dlatrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(8, dtrt03(uplo, trans, diag, n, 1, a.Matrix(lda, opts), scale, rwork, one, b.MatrixOff(n, lda, opts), x.Matrix(lda, opts), work))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATRS"), uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail++
					}
					if result.Get(8) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATRS"), uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail++
					}
					nrun += 2
				}
			}
		label110:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 7672
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
