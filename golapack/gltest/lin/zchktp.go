package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zchktp tests Ztptri, -TRS, -RFS, and -CON, and Zlatps
func zchktp(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, ap, ainvp, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var norm, xtype byte
	var diag mat.MatDiag
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, imat, in, info, irhs, k, lap, lda, n, nerrs, nfail, nrhs, nrun, ntype1, ntypes int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntype1 = 10
	ntypes = 18
	one = 1.0
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Ztp"
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
		lap = lda * (lda + 1) / 2
		xtype = 'N'

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label70
			}

			for _, uplo = range mat.IterMatUplo(false) {
				//              Do first for uplo='U', then for uplo='L'

				//              Call Zlattp to generate a triangular test matrix.
				*srnamt = "Zlattp"
				diag = zlattp(imat, uplo, NoTrans, &iseed, n, ap, x, work, rwork)

				//              Set IDIAG = 1 for non-unit matrices, 2 for unit.
				if diag == NonUnit {
					idiag = 1
				} else {
					idiag = 2
				}

				//+    TEST 1
				//              Form the inverse of A.
				if n > 0 {
					goblas.Zcopy(lap, ap.Off(0, 1), ainvp.Off(0, 1))
				}
				*srnamt = "Ztptri"
				if info, err = golapack.Ztptri(uplo, diag, n, ainvp); err != nil {
					panic(err)
				}

				//              Check error code from Ztptri.
				if info != 0 {
					t.Fail()
					nerrs = alaerh(path, "Ztptri", info, 0, []byte{uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				//              Compute the infinity-norm condition number of A.
				anorm = golapack.Zlantp('I', uplo, diag, n, ap, rwork)
				ainvnm = golapack.Zlantp('I', uplo, diag, n, ainvp, rwork)
				if anorm <= zero || ainvnm <= zero {
					rcondi = one
				} else {
					rcondi = (one / anorm) / ainvnm
				}

				//              Compute the residual for the triangular matrix times its
				//              inverse.  Also compute the 1-norm condition number of A.
				rcondo, *result.GetPtr(0) = ztpt01(uplo, diag, n, ap, ainvp, rwork)

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(0) >= thresh {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					fmt.Printf(" uplo=%s, diag=%s, n=%5d, _type %2d, test(%2d)= %12.5f\n", uplo, diag, n, imat, 1, result.Get(0))
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
						*srnamt = "zlarhs"
						if err = zlarhs(path, xtype, uplo, trans, n, n, 0, idiag, nrhs, ap.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
							panic(err)
						}
						xtype = 'C'
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "Ztptrs"
						if info, err = golapack.Ztptrs(uplo, trans, diag, n, nrhs, ap, x.CMatrix(lda, opts)); err != nil || info != 0 {
							t.Fail()
							nerrs = alaerh(path, "Ztptrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						*result.GetPtr(1) = ztpt02(uplo, trans, diag, n, nrhs, ap, x.CMatrix(lda, opts), b.CMatrix(lda, opts), work, rwork)

						//+    TEST 3
						//                 Check solution from generated exact solution.
						*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

						//+    TESTS 4, 5, and 6
						//                 Use iterative refinement to improve the solution and
						//                 compute error bounds.
						*srnamt = "Ztprfs"
						if err = golapack.Ztprfs(uplo, trans, diag, n, nrhs, ap, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil || info != 0 {
							t.Fail()
							nerrs = alaerh(path, "Ztprfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
						}

						*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
						ztpt05(uplo, trans, diag, n, nrhs, ap, b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= 6; k++ {
							if result.Get(k-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								fmt.Printf(" uplo=%s, trans=%s, diag=%s, n=%5d', nrhs=%5d, _type %2d, test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
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
					*srnamt = "Ztpcon"
					if rcond, err = golapack.Ztpcon(norm, uplo, diag, n, ap, work, rwork); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Ztpcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					*result.GetPtr(6) = ztpt06(rcond, rcondc, uplo, diag, n, ap, rwork)

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(6) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" %s( '%c', %s, %s,%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "Ztpcon", norm, uplo, diag, n, imat, 7, result.Get(6))
						nfail++
					}
					nrun++
				}
			}
		label70:
		}

		//        Use pathological test matrices to test Zlatps.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label100
			}

			for _, uplo = range mat.IterMatUplo(false) {

				for _, trans = range mat.IterMatTrans() {

					//                 Call Zlattp to generate a triangular test matrix.
					*srnamt = "Zlattp"
					diag = zlattp(imat, uplo, trans, &iseed, n, ap, x, work, rwork)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "Zlatps"
					goblas.Zcopy(n, x.Off(0, 1), b.Off(0, 1))
					if scale, err = golapack.Zlatps(uplo, trans, diag, 'N', n, ap, b, rwork); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatps", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					*result.GetPtr(7) = ztpt03(uplo, trans, diag, n, 1, ap, scale, rwork, one, b.CMatrix(lda, opts), x.CMatrix(lda, opts), work)

					//+    TEST 9
					//                 Solve op(A)*x = b again with NORMIN = 'Y'.
					goblas.Zcopy(n, x.Off(0, 1), b.Off(n, 1))
					if scale, err = golapack.Zlatps(uplo, trans, diag, 'Y', n, ap, b.Off(n), rwork); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatps", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					*result.GetPtr(8) = ztpt03(uplo, trans, diag, n, 1, ap, scale, rwork, one, b.CMatrixOff(n, lda, opts), x.CMatrix(lda, opts), work)

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "Zlatps", uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail++
					}
					if result.Get(8) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "Zlatps", uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail++
					}
					nrun += 2
				}
			}
		label100:
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
