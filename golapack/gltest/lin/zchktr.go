package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zchktr tests Ztrtri, -TRS, -RFS, and -CON, and Zlatrs
func zchktr(dotype []bool, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var norm, xtype byte
	var diag mat.MatDiag
	var trans mat.MatTrans
	var uplo mat.MatUplo
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, imat, in, inb, info, irhs, k, lda, n, nb, nerrs, nfail, nrhs, nrun, ntype1, ntypes int
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
	path := "Ztr"
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

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label80
			}

			for _, uplo = range mat.IterMatUplo(false) {

				//              Call Zlattr to generate a triangular test matrix.
				*srnamt = "Zlattr"
				diag = zlattr(imat, uplo, NoTrans, &iseed, n, a.CMatrix(lda, opts), x, work, rwork)

				//              Set IDIAG = 1 for non-unit matrices, 2 for unit.
				if diag == NonUnit {
					idiag = 1
				} else {
					idiag = 2
				}

				for inb = 1; inb <= nnb; inb++ {
					//                 Do for each blocksize in NBVAL
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//+    TEST 1
					//                 Form the inverse of A.
					golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
					*srnamt = "Ztrtri"
					if info, err = golapack.Ztrtri(uplo, diag, n, ainv.CMatrix(lda, opts)); err != nil || info != 0 {
						t.Fail()
						nerrs = alaerh(path, "Ztrtri", info, 0, []byte{uplo.Byte(), diag.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Zlantr('I', uplo, diag, n, n, a.CMatrix(lda, opts), rwork)
					ainvnm = golapack.Zlantr('I', uplo, diag, n, n, ainv.CMatrix(lda, opts), rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					//                 Compute the residual for the triangular matrix times
					//                 its inverse.  Also compute the 1-norm condition number
					//                 of A.
					rcondo, *result.GetPtr(0) = ztrt01(uplo, diag, n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts), rwork)
					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(0) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" uplo=%s, diag=%s, n=%5d, NB=%4d, _type %2d, test(%2d)= %12.5f\n", uplo, diag, n, nb, imat, 1, result.Get(0))
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
							*srnamt = "zlarhs"
							if err = zlarhs(path, xtype, uplo, trans, n, n, 0, idiag, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

							*srnamt = "Ztrtrs"
							if info, err = golapack.Ztrtrs(uplo, trans, diag, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts)); err != nil || info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Ztrtrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							*result.GetPtr(1) = ztrt02(uplo, trans, diag, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), b.CMatrix(lda, opts), work, rwork)

							//+    TEST 3
							//                       Check solution from generated exact solution.
							*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

							//+    TESTS 4, 5, and 6
							//                       Use iterative refinement to improve the solution
							//                       and compute error bounds.
							*srnamt = "Ztrrfs"
							if err = golapack.Ztrrfs(uplo, trans, diag, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
								t.Fail()
								nerrs = alaerh(path, "Ztrrfs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}
							//
							*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							ztrt05(uplo, trans, diag, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" uplo=%s, trans=%s, diag=%s, n=%5d, nb=%4d, _type %2d,        test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
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
						*srnamt = "Ztrcon"
						if rcond, err = golapack.Ztrcon(norm, uplo, diag, n, a.CMatrix(lda, opts), work, rwork); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Ztrcon", info, 0, []byte{norm, uplo.Byte(), diag.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}

						*result.GetPtr(6) = ztrt06(rcond, rcondc, uplo, diag, n, a.CMatrix(lda, opts), rwork)

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" norm='%c', uplo=%s, n=%5d,            _type %2d, test(%2d)=%12.5f\n", norm, uplo, n, imat, 7, result.Get(6))
							nfail++
						}
						nrun++
					}
				label60:
				}
			}
		label80:
		}

		//        Use pathological test matrices to test Zlatrs.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label110
			}

			for _, uplo = range mat.IterMatUplo(false) {

				for _, trans = range mat.IterMatTrans() {

					//                 Call Zlattr to generate a triangular test matrix.
					*srnamt = "Zlattr"
					diag = zlattr(imat, uplo, trans, &iseed, n, a.CMatrix(lda, opts), x, work, rwork)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "Zlatrs"
					goblas.Zcopy(n, x.Off(0, 1), b.Off(0, 1))
					if scale, err = golapack.Zlatrs(uplo, trans, diag, 'N', n, a.CMatrix(lda, opts), b, rwork); err != nil {
						panic(err)
					}

					//                 Check error code from Zlatrs.
					if info != 0 {
						t.Fail()
						nerrs = alaerh(path, "Zlatrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'N'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					*result.GetPtr(7) = ztrt03(uplo, trans, diag, n, 1, a.CMatrix(lda, opts), scale, rwork, one, b.CMatrix(lda, opts), x.CMatrix(lda, opts), work)

					//+    TEST 9
					//                 Solve op(A)*X = b again with NORMIN = 'Y'.
					goblas.Zcopy(n, x.Off(0, 1), b.Off(n, 1))
					if scale, err = golapack.Zlatrs(uplo, trans, diag, 'Y', n, a.CMatrix(lda, opts), b.Off(n), rwork); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatrs", info, 0, []byte{uplo.Byte(), trans.Byte(), diag.Byte(), 'Y'}, n, n, -1, -1, -1, imat, nfail, nerrs)
					}

					*result.GetPtr(8) = ztrt03(uplo, trans, diag, n, 1, a.CMatrix(lda, opts), scale, rwork, one, b.CMatrixOff(n, lda, opts), x.CMatrix(lda, opts), work)

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "Zlatrs", uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail++
					}
					if result.Get(8) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" %s( %s, %s, %s, '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "Zlatrs", uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail++
					}
					nrun += 2
				}
			}
		label110:
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
