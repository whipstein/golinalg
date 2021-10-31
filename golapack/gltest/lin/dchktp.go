package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dchktp tests DTPTRI, -TRS, -RFS, and -CON, and DLATPS
func dchktp(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, ap, ainvp, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var norm, xtype byte
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var diag mat.MatDiag
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var _result *float64
	var i, imat, in, info, irhs, k, lap, lda, n, nerrs, nfail, nrhs, nrun, ntype1, ntypes int
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
	path := "Dtp"
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
		lap = lda * (lda + 1) / 2
		xtype = 'N'

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label70
			}

			for _, uplo = range mat.IterMatUplo(false) {
				//              Do first for uplo = 'U', then for uplo = 'L'

				//              Call DLATTP to generate a triangular test matrix.
				*srnamt = "Dlattp"
				diag, iseed, err = dlattp(imat, uplo, NoTrans, iseed, n, ap, x, work)

				//+    TEST 1
				//              Form the inverse of A.
				if n > 0 {
					goblas.Dcopy(lap, ap.Off(0, 1), ainvp.Off(0, 1))
				}
				*srnamt = "Dtptri"
				if info, err = golapack.Dtptri(uplo, diag, n, ainvp); err != nil || info != 0 {
					nerrs = alaerh(path, "Dtptri", info, 0, []byte{uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				//              Compute the infinity-norm condition number of A.
				anorm = golapack.Dlantp('I', uplo, diag, n, ap, rwork)
				ainvnm = golapack.Dlantp('I', uplo, diag, n, ainvp, rwork)
				if anorm <= zero || ainvnm <= zero {
					rcondi = one
				} else {
					rcondi = (one / anorm) / ainvnm
				}

				//              Compute the residual for the triangular matrix times its
				//              inverse.  Also compute the 1-norm condition number of A.
				_result = result.GetPtr(0)
				rcondo, *_result = dtpt01(uplo, diag, n, ap, ainvp, rwork)

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(0) >= thresh {
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					t.Fail()
					fmt.Printf(" uplo=%s, diag=%s, n=%5d, type %2d, test(%2d)= %12.5f\n", uplo, diag, n, imat, 1, result.Get(0))
					nfail++
				}
				nrun++

				for irhs = 1; irhs <= nns; irhs++ {
					nrhs = nsval[irhs-1]
					xtype = 'N'

					for _, trans = range mat.IterMatTrans() {
						//                 Do for op(A) = A, A**T, or A**H.
						if trans == NoTrans {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}

						//+    TEST 2
						//                 Solve and compute residual for op(A)*x = b.
						*srnamt = "Dlarhs"
						if err = Dlarhs(path, xtype, uplo, trans, n, n, 0, int(diag)+1, nrhs, ap.Matrix(lap, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						xtype = 'C'
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

						*srnamt = "Dtptrs"
						if info, err = golapack.Dtptrs(uplo, trans, diag, n, nrhs, ap, x.Matrix(lda, opts)); err != nil || info != 0 {
							nerrs = alaerh(path, "Dtptrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						result.Set(1, dtpt02(uplo, trans, diag, n, nrhs, ap, x.Matrix(lda, opts), b.Matrix(lda, opts), work))

						//+    TEST 3
						//                 Check solution from generated exact solution.
						result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

						//+    TESTS 4, 5, and 6
						//                 Use iterative refinement to improve the solution and
						//                 compute error bounds.
						*srnamt = "Dtprfs"
						if err = golapack.Dtprfs(uplo, trans, diag, n, nrhs, ap, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, &iwork); err != nil {
							nerrs = alaerh(path, "Dtprfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
						dtpt05(mat.MatUplo(uplo), mat.MatTrans(trans), diag, n, nrhs, ap, b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= 6; k++ {
							if result.Get(k-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								t.Fail()
								fmt.Printf(" uplo=%s, trans=%s, diag=%s, n=%5d', nrhs=%5d, type %2d, test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun += 5
					}
				}

				//+    TEST 7
				//                 Get an estimate of RCOND = 1/CNDNUM.
				for _, trans = range mat.IterMatTrans(false) {
					if trans == NoTrans {
						norm = 'O'
						rcondc = rcondo
					} else {
						norm = 'I'
						rcondc = rcondi
					}

					*srnamt = "Dtpcon"
					if rcond, err = golapack.Dtpcon(norm, uplo, diag, n, ap, work, &iwork); err != nil {
						nerrs = alaerh(path, "Dtpcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(6, dtpt06(rcond, rcondc, uplo, diag, n, ap, rwork))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(6) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', %s, %s,%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DTPCON"), norm, uplo, diag, n, imat, 7, result.Get(6))
						nfail++
					}
					nrun++
				}
			}
		label70:
		}

		//        Use pathological test matrices to test DLATPS.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label100
			}

			for _, uplo = range mat.IterMatUplo(false) {
				//              Do first for uplo = 'U', then for uplo = 'L'
				for _, trans = range mat.IterMatTrans() {
					//                 Do for op(A) = A, A**T, or A**H.

					//                 Call DLATTP to generate a triangular test matrix.
					*srnamt = "Dlattp"
					diag, iseed, err = dlattp(imat, uplo, trans, iseed, n, ap, x, work)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "Dlatps"
					goblas.Dcopy(n, x.Off(0, 1), b.Off(0, 1))
					if scale, err = golapack.Dlatps(uplo, trans, diag, 'N', n, ap, b, rwork); err != nil {
						panic(err)
					}

					//                 Check error code from DLATPS.
					if info != 0 {
						nerrs = alaerh(path, "Dlatps", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(7, dtpt03(uplo, trans, diag, n, 1, ap, scale, rwork, one, b.Matrix(lda, opts), x.Matrix(lda, opts), work))

					//+    TEST 9
					//                 Solve op(A)*x = b again with NORMIN = 'Y'.
					goblas.Dcopy(n, x.Off(0, 1), b.Off(n, 1))
					if scale, err = golapack.Dlatps(uplo, trans, diag, 'Y', n, ap, b.Off(n), rwork); err != nil {
						panic(err)
					}

					//                 Check error code from DLATPS.
					if info != 0 {
						nerrs = alaerh(path, "Dlatps", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					result.Set(8, dtpt03(uplo, trans, diag, n, 1, ap, scale, rwork, one, b.MatrixOff(n, lda, opts), x.Matrix(lda, opts), work))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATPS"), uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail++
					}
					if result.Get(8) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATPS"), uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail++
					}
					nrun += 2
				}
			}
		label100:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 7392
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
