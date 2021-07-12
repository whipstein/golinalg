package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zchktr tests ZTRTRI, -TRS, -RFS, and -CON, and ZLATRS
func Zchktr(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var diag, norm, trans, uplo, xtype byte
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, imat, in, inb, info, irhs, itran, iuplo, k, lda, n, nb, nerrs, nfail, nrhs, nrun, ntran, ntype1, ntypes int

	transs := make([]byte, 3)
	uplos := make([]byte, 2)
	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntype1 = 10
	ntypes = 18
	ntran = 3
	one = 1.0
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], transs[0], transs[1], transs[2] = 'U', 'L', 'N', 'T', 'C'

	//     Initialize constants and the random number seed.
	path := []byte("ZTR")
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
		lda = max(1, n)
		xtype = 'N'

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label80
			}

			for iuplo = 1; iuplo <= 2; iuplo++ {
				//              Do first for UPLO = 'U', then for UPLO = 'L'
				uplo = uplos[iuplo-1]

				//              Call ZLATTR to generate a triangular test matrix.
				*srnamt = "ZLATTR"
				Zlattr(&imat, uplo, 'N', &diag, &iseed, &n, a.CMatrix(lda, opts), &lda, x, work, rwork, &info)

				//              Set IDIAG = 1 for non-unit matrices, 2 for unit.
				if diag == 'N' {
					idiag = 1
				} else {
					idiag = 2
				}

				for inb = 1; inb <= (*nnb); inb++ {
					//                 Do for each blocksize in NBVAL
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//+    TEST 1
					//                 Form the inverse of A.
					golapack.Zlacpy(uplo, &n, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda)
					*srnamt = "ZTRTRI"
					golapack.Ztrtri(uplo, diag, &n, ainv.CMatrix(lda, opts), &lda, &info)

					//                 Check error code from ZTRTRI.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZTRTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Zlantr('I', uplo, diag, &n, &n, a.CMatrix(lda, opts), &lda, rwork)
					ainvnm = golapack.Zlantr('I', uplo, diag, &n, &n, ainv.CMatrix(lda, opts), &lda, rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					//                 Compute the residual for the triangular matrix times
					//                 its inverse.  Also compute the 1-norm condition number
					//                 of A.
					Ztrt01(uplo, diag, &n, a.CMatrix(lda, opts), &lda, ainv.CMatrix(lda, opts), &lda, &rcondo, rwork, result.GetPtr(0))
					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(0) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" UPLO='%c', DIAG='%c', N=%5d, NB=%4d, _type %2d, test(%2d)= %12.5f\n", uplo, diag, n, nb, imat, 1, result.Get(0))
						nfail = nfail + 1
					}
					nrun = nrun + 1

					//                 Skip remaining tests if not the first block size.
					if inb != 1 {
						goto label60
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

							//+    TEST 2
							//                       Solve and compute residual for op(A)*x = b.
							*srnamt = "ZLARHS"
							Zlarhs(path, xtype, uplo, trans, &n, &n, func() *int { y := 0; return &y }(), &idiag, &nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)
							xtype = 'C'
							golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

							*srnamt = "ZTRTRS"
							golapack.Ztrtrs(uplo, trans, diag, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &info)

							//                       Check error code from ZTRTRS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZTRTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}

							Ztrt02(uplo, trans, diag, &n, &nrhs, a.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, work, rwork, result.GetPtr(1))

							//+    TEST 3
							//                       Check solution from generated exact solution.
							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

							//+    TESTS 4, 5, and 6
							//                       Use iterative refinement to improve the solution
							//                       and compute error bounds.
							*srnamt = "ZTRRFS"
							golapack.Ztrrfs(uplo, trans, diag, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs), &info)

							//                       Check error code from ZTRRFS.
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZTRRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}
							//
							Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
							Ztrt05(uplo, trans, diag, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" UPLO='%c', TRANS='%c', DIAG='%c', N=%5d, NB=%4d, _type %2d,        test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 5
						}
					}

					//+    TEST 7
					//                       Get an estimate of RCOND = 1/CNDNUM.
					for itran = 1; itran <= 2; itran++ {
						if itran == 1 {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}
						*srnamt = "ZTRCON"
						golapack.Ztrcon(norm, uplo, diag, &n, a.CMatrix(lda, opts), &lda, &rcond, work, rwork, &info)

						//                       Check error code from ZTRCON.
						if info != 0 {
							t.Fail()
							Alaerh(path, []byte("ZTRCON"), &info, func() *int { y := 0; return &y }(), []byte{norm, uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						Ztrt06(&rcond, &rcondc, uplo, diag, &n, a.CMatrix(lda, opts), &lda, rwork, result.GetPtr(6))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" NORM='%c', UPLO ='%c', N=%5d,            _type %2d, test(%2d)=%12.5f\n", norm, uplo, n, imat, 7, result.Get(6))
							nfail = nfail + 1
						}
						nrun = nrun + 1
					}
				label60:
				}
			}
		label80:
		}

		//        Use pathological test matrices to test ZLATRS.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label110
			}

			for iuplo = 1; iuplo <= 2; iuplo++ {
				//              Do first for UPLO = 'U', then for UPLO = 'L'
				uplo = uplos[iuplo-1]
				for itran = 1; itran <= ntran; itran++ {
					//                 Do for op(A) = A, A**T, and A**H.
					trans = transs[itran-1]

					//                 Call ZLATTR to generate a triangular test matrix.
					*srnamt = "ZLATTR"
					Zlattr(&imat, uplo, trans, &diag, &iseed, &n, a.CMatrix(lda, opts), &lda, x, work, rwork, &info)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "ZLATRS"
					goblas.Zcopy(n, x.Off(0, 1), b.Off(0, 1))
					golapack.Zlatrs(uplo, trans, diag, 'N', &n, a.CMatrix(lda, opts), &lda, b, &scale, rwork, &info)

					//                 Check error code from ZLATRS.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZLATRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'N'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Ztrt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), a.CMatrix(lda, opts), &lda, &scale, rwork, &one, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, work, result.GetPtr(7))

					//+    TEST 9
					//                 Solve op(A)*X = b again with NORMIN = 'Y'.
					goblas.Zcopy(n, x.Off(0, 1), b.Off(n, 1))
					golapack.Zlatrs(uplo, trans, diag, 'Y', &n, a.CMatrix(lda, opts), &lda, b.Off(n), &scale, rwork, &info)

					//                 Check error code from ZLATRS.
					if info != 0 {
						t.Fail()
						Alaerh(path, []byte("ZLATRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'Y'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Ztrt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), a.CMatrix(lda, opts), &lda, &scale, rwork, &one, b.CMatrixOff(n, lda, opts), &lda, x.CMatrix(lda, opts), &lda, work, result.GetPtr(8))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "ZLATRS", uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail = nfail + 1
					}
					if result.Get(8) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), _type %2d, test(%2d)=%12.5f\n", "ZLATRS", uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail = nfail + 1
					}
					nrun = nrun + 2
				}
			}
		label110:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
