package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dchktr tests DTRTRI, -TRS, -RFS, and -CON, and DLATRS
func Dchktr(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, a, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var diag, norm, trans, uplo, xtype byte
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, imat, in, inb, info, irhs, itran, iuplo, k, lda, n, nb, nerrs, nfail, nrhs, nrun, ntran, ntype1, ntypes int

	transs := make([]byte, 3)
	uplos := make([]byte, 2)
	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	ntype1 = 10
	ntypes = 18
	ntran = 3
	one = 1.0
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1], transs[0], transs[1], transs[2] = 'U', 'L', 'N', 'T', 'C'

	//     Initialize constants and the random number seed.
	path := []byte("DTR")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrtr(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

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

				//              Call DLATTR to generate a triangular test matrix.
				*srnamt = "DLATTR"
				Dlattr(&imat, uplo, 'N', &diag, &iseed, &n, a.Matrix(lda, opts), &lda, x, work, &info)

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
					golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
					*srnamt = "DTRTRI"
					golapack.Dtrtri(uplo, diag, &n, ainv.Matrix(lda, opts), &lda, &info)
					//
					//                 Check error code from DTRTRI.
					//
					if info != 0 {
						Alaerh(path, []byte("DTRTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}

					//                 Compute the infinity-norm condition number of A.
					anorm = golapack.Dlantr('I', uplo, diag, &n, &n, a.Matrix(lda, opts), &lda, rwork)
					ainvnm = golapack.Dlantr('I', uplo, diag, &n, &n, ainv.Matrix(lda, opts), &lda, rwork)
					if anorm <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anorm) / ainvnm
					}

					//                 Compute the residual for the triangular matrix times
					//                 its inverse.  Also compute the 1-norm condition number
					//                 of A.
					Dtrt01(uplo, diag, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, &rcondo, rwork, result.GetPtr(0))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(0) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" UPLO='%c', DIAG='%c', N=%5d, NB=%4d, type %2d, test(%2d)= %12.5f\n", uplo, diag, n, nb, imat, 1, result.Get(0))
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
							*srnamt = "DLARHS"
							Dlarhs(path, &xtype, uplo, trans, &n, &n, func() *int { y := 0; return &y }(), &idiag, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
							xtype = 'C'
							golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

							*srnamt = "DTRTRS"
							golapack.Dtrtrs(uplo, trans, diag, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &info)

							//                       Check error code from DTRTRS.
							if info != 0 {
								Alaerh(path, []byte("DTRTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}

							Dtrt02(uplo, trans, diag, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, work, result.GetPtr(1))

							//+    TEST 3
							//                       Check solution from generated exact solution.
							Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

							//+    TESTS 4, 5, and 6
							//                       Use iterative refinement to improve the solution
							//                       and compute error bounds.
							*srnamt = "DTRRFS"
							golapack.Dtrrfs(uplo, trans, diag, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), work, iwork, &info)

							//                       Check error code from DTRRFS.
							if info != 0 {
								Alaerh(path, []byte("DTRRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}

							Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
							Dtrt05(uplo, trans, diag, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), result.Off(4))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 2; k <= 6; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									t.Fail()
									fmt.Printf(" UPLO='%c', TRANS='%c', DIAG='%c', N=%5d, NB=%4d, type %2d,        test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
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
						*srnamt = "DTRCON"
						golapack.Dtrcon(norm, uplo, diag, &n, a.Matrix(lda, opts), &lda, &rcond, work, iwork, &info)

						//                       Check error code from DTRCON.
						if info != 0 {
							Alaerh(path, []byte("DTRCON"), &info, func() *int { y := 0; return &y }(), []byte{norm, uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						Dtrt06(&rcond, &rcondc, uplo, diag, &n, a.Matrix(lda, opts), &lda, rwork, result.GetPtr(6))

						//                    Print the test ratio if it is .GE. THRESH.
						if result.Get(6) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
							fmt.Printf(" NORM='%c', UPLO ='%c', N=%5d,            type %2d, test(%2d)=%12.5f\n", norm, uplo, n, imat, 7, result.Get(6))
							nfail = nfail + 1
						}
						nrun = nrun + 1
					}
				label60:
				}
			}
		label80:
		}

		//        Use pathological test matrices to test DLATRS.
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

					//                 Call DLATTR to generate a triangular test matrix.
					*srnamt = "DLATTR"
					Dlattr(&imat, uplo, trans, &diag, &iseed, &n, a.Matrix(lda, opts), &lda, x, work, &info)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "DLATRS"
					goblas.Dcopy(n, x.Off(0, 1), b.Off(0, 1))
					golapack.Dlatrs(uplo, trans, diag, 'N', &n, a.Matrix(lda, opts), &lda, b, &scale, rwork, &info)

					//                 Check error code from DLATRS.
					if info != 0 {
						Alaerh(path, []byte("DLATRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'N'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dtrt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), a.Matrix(lda, opts), &lda, &scale, rwork, &one, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work, result.GetPtr(7))

					//+    TEST 9
					//                 Solve op(A)*X = b again with NORMIN = 'Y'.
					goblas.Dcopy(n, x.Off(0, 1), b.Off(n, 1))
					golapack.Dlatrs(uplo, trans, diag, 'Y', &n, a.Matrix(lda, opts), &lda, b.Off(n), &scale, rwork, &info)

					//                 Check error code from DLATRS.
					if info != 0 {
						Alaerh(path, []byte("DLATRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'Y'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dtrt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), a.Matrix(lda, opts), &lda, &scale, rwork, &one, b.MatrixOff(n, lda, opts), &lda, x.Matrix(lda, opts), &lda, work, result.GetPtr(8))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATRS"), uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail = nfail + 1
					}
					if result.Get(8) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATRS"), uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail = nfail + 1
					}
					nrun = nrun + 2
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
	Alasum(path, &nfail, &nrun, &nerrs)
}
