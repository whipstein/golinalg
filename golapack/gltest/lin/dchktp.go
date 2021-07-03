package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dchktp tests DTPTRI, -TRS, -RFS, and -CON, and DLATPS
func Dchktp(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, nmax *int, ap, ainvp, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var diag, norm, trans, uplo, xtype byte
	var ainvnm, anorm, one, rcond, rcondc, rcondi, rcondo, scale, zero float64
	var i, idiag, imat, in, info, irhs, itran, iuplo, k, lap, lda, n, nerrs, nfail, nrhs, nrun, ntran, ntype1, ntypes int

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
	path := []byte("DTP")
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

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL
		n = (*nval)[in-1]
		lda = maxint(1, n)
		lap = lda * (lda + 1) / 2
		xtype = 'N'

		for imat = 1; imat <= ntype1; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label70
			}

			for iuplo = 1; iuplo <= 2; iuplo++ {
				//              Do first for UPLO = 'U', then for UPLO = 'L'
				uplo = uplos[iuplo-1]

				//              Call DLATTP to generate a triangular test matrix.
				*srnamt = "DLATTP"
				Dlattp(&imat, uplo, 'N', &diag, &iseed, &n, ap, x, work, &info)

				//              Set IDIAG = 1 for non-unit matrices, 2 for unit.
				if diag == 'N' {
					idiag = 1
				} else {
					idiag = 2
				}

				//+    TEST 1
				//              Form the inverse of A.
				if n > 0 {
					goblas.Dcopy(lap, ap, 1, ainvp, 1)
				}
				*srnamt = "DTPTRI"
				golapack.Dtptri(uplo, diag, &n, ainvp, &info)

				//              Check error code from DTPTRI.
				if info != 0 {
					Alaerh(path, []byte("DTPTRI"), &info, func() *int { y := 0; return &y }(), []byte{uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				//              Compute the infinity-norm condition number of A.
				anorm = golapack.Dlantp('I', uplo, diag, &n, ap, rwork)
				ainvnm = golapack.Dlantp('I', uplo, diag, &n, ainvp, rwork)
				if anorm <= zero || ainvnm <= zero {
					rcondi = one
				} else {
					rcondi = (one / anorm) / ainvnm
				}

				//              Compute the residual for the triangular matrix times its
				//              inverse.  Also compute the 1-norm condition number of A.
				Dtpt01(uplo, diag, &n, ap, ainvp, &rcondo, rwork, result.GetPtr(0))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(0) >= (*thresh) {
					if nfail == 0 && nerrs == 0 {
						Alahd(path)
					}
					t.Fail()
					fmt.Printf(" UPLO='%c', DIAG='%c', N=%5d, type %2d, test(%2d)= %12.5f\n", uplo, diag, n, imat, 1, result.Get(0))
					nfail = nfail + 1
				}
				nrun = nrun + 1

				for irhs = 1; irhs <= (*nns); irhs++ {
					nrhs = (*nsval)[irhs-1]
					xtype = 'N'

					for itran = 1; itran <= ntran; itran++ {
						//                 Do for op(A) = A, A**T, or A**H.
						trans = transs[itran-1]
						if itran == 1 {
							norm = 'O'
							rcondc = rcondo
						} else {
							norm = 'I'
							rcondc = rcondi
						}

						//+    TEST 2
						//                 Solve and compute residual for op(A)*x = b.
						*srnamt = "DLARHS"
						Dlarhs(path, &xtype, uplo, trans, &n, &n, func() *int { y := 0; return &y }(), &idiag, &nrhs, ap.Matrix(lap, opts), &lap, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
						xtype = 'C'
						golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						*srnamt = "DTPTRS"
						golapack.Dtptrs(uplo, trans, diag, &n, &nrhs, ap, x.Matrix(lda, opts), &lda, &info)

						//                 Check error code from DTPTRS.
						if info != 0 {
							Alaerh(path, []byte("DTPTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						Dtpt02(uplo, trans, diag, &n, &nrhs, ap, x.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, work, result.GetPtr(1))

						//+    TEST 3
						//                 Check solution from generated exact solution.
						Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

						//+    TESTS 4, 5, and 6
						//                 Use iterative refinement to improve the solution and
						//                 compute error bounds.
						*srnamt = "DTPRFS"
						golapack.Dtprfs(uplo, trans, diag, &n, &nrhs, ap, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, iwork, &info)

						//                 Check error code from DTPRFS.
						if info != 0 {
							Alaerh(path, []byte("DTPRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
						}

						Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
						Dtpt05(uplo, trans, diag, &n, &nrhs, ap, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(4))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= 6; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								t.Fail()
								fmt.Printf(" UPLO='%c', TRANS='%c', DIAG='%c', N=%5d', NRHS=%5d, type %2d, test(%2d)= %12.5f\n", uplo, trans, diag, n, nrhs, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + 5
					}
				}

				//+    TEST 7
				//                 Get an estimate of RCOND = 1/CNDNUM.
				for itran = 1; itran <= 2; itran++ {
					if itran == 1 {
						norm = 'O'
						rcondc = rcondo
					} else {
						norm = 'I'
						rcondc = rcondi
					}

					*srnamt = "DTPCON"
					golapack.Dtpcon(norm, uplo, diag, &n, ap, &rcond, work, iwork, &info)

					//                 Check error code from DTPCON.
					if info != 0 {
						Alaerh(path, []byte("DTPCON"), &info, func() *int { y := 0; return &y }(), []byte{norm, uplo, diag}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dtpt06(&rcond, &rcondc, uplo, diag, &n, ap, rwork, result.GetPtr(6))

					//                 Print the test ratio if it is .GE. THRESH.
					if result.Get(6) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', '%c', '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DTPCON"), norm, uplo, diag, n, imat, 7, result.Get(6))
						nfail = nfail + 1
					}
					nrun = nrun + 1
				}
			}
		label70:
		}

		//        Use pathological test matrices to test DLATPS.
		for imat = ntype1 + 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label100
			}

			for iuplo = 1; iuplo <= 2; iuplo++ {
				//              Do first for UPLO = 'U', then for UPLO = 'L'
				uplo = uplos[iuplo-1]
				for itran = 1; itran <= ntran; itran++ {
					//                 Do for op(A) = A, A**T, or A**H.
					trans = transs[itran-1]

					//                 Call DLATTP to generate a triangular test matrix.
					*srnamt = "DLATTP"
					Dlattp(&imat, uplo, trans, &diag, &iseed, &n, ap, x, work, &info)

					//+    TEST 8
					//                 Solve the system op(A)*x = b.
					*srnamt = "DLATPS"
					goblas.Dcopy(n, x, 1, b, 1)
					golapack.Dlatps(uplo, trans, diag, 'N', &n, ap, b, &scale, rwork, &info)

					//                 Check error code from DLATPS.
					if info != 0 {
						Alaerh(path, []byte("DLATPS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'N'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dtpt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), ap, &scale, rwork, &one, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work, result.GetPtr(7))

					//+    TEST 9
					//                 Solve op(A)*x = b again with NORMIN = 'Y'.
					goblas.Dcopy(n, x, 1, b.Off(n+1-1), 1)
					golapack.Dlatps(uplo, trans, diag, 'Y', &n, ap, b.Off(n+1-1), &scale, rwork, &info)

					//                 Check error code from DLATPS.
					if info != 0 {
						Alaerh(path, []byte("DLATPS"), &info, func() *int { y := 0; return &y }(), []byte{uplo, trans, diag, 'Y'}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					}

					Dtpt03(uplo, trans, diag, &n, func() *int { y := 1; return &y }(), ap, &scale, rwork, &one, b.MatrixOff(n+1-1, lda, opts), &lda, x.Matrix(lda, opts), &lda, work, result.GetPtr(8))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					if result.Get(7) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATPS"), uplo, trans, diag, 'N', n, imat, 8, result.Get(7))
						nfail = nfail + 1
					}
					if result.Get(8) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" %s( '%c', '%c', '%c', '%c',%5d, ... ), type %2d, test(%2d)=%12.5f\n", []byte("DLATPS"), uplo, trans, diag, 'Y', n, imat, 9, result.Get(8))
						nfail = nfail + 1
					}
					nrun = nrun + 2
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
	Alasum(path, &nfail, &nrun, &nerrs)
}
