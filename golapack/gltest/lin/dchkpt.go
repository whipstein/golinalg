package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dchkpt tests DPTTRF, -TRS, -RFS, and -CON
func Dchkpt(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, a, d, e, b, x, xact, work, rwork *mat.Vector, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type byte
	var ainvnm, anorm, cond, dmax, one, rcond, rcondc, zero float64
	var i, ia, imat, in, info, irhs, ix, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int

	result := vf(7)
	z := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 12
	// ntests = 7

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 0, 0, 0, 1

	path := []byte("DPT")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrgt(path, t)
	}
	(*infot) = 0

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL.
		n = (*nval)[in-1]
		lda = max(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if n > 0 && !(*dotype)[imat-1] {
				goto label100
			}

			//           Set up parameters with DLATB4.
			Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cond, &dist)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Type 1-6:  generate a symmetric tridiagonal matrix of
				//              known condition number in lower triangular band storage.
				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cond, &anorm, &kl, &ku, 'B', a.Matrix(2, opts), func() *int { y := 2; return &y }(), work, &info)

				//              Check the error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
					goto label100
				}
				izero = 0

				//              Copy the matrix to D and E.
				ia = 1
				for i = 1; i <= n-1; i++ {
					d.Set(i-1, a.Get(ia-1))
					e.Set(i-1, a.Get(ia))
					ia = ia + 2
				}
				if n > 0 {
					d.Set(n-1, a.Get(ia-1))
				}
			} else {
				//              Type 7-12:  generate a diagonally dominant matrix with
				//              unknown condition number in the vectors D and E.
				if !zerot || !(*dotype)[6] {
					//                 Let D and E have values from [-1,1].
					golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &n, d)
					golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(n-1), e)

					//                 Make the tridiagonal matrix diagonally dominant.
					if n == 1 {
						d.Set(0, math.Abs(d.Get(0)))
					} else {
						d.Set(0, math.Abs(d.Get(0))+math.Abs(e.Get(0)))
						d.Set(n-1, math.Abs(d.Get(n-1))+math.Abs(e.Get(n-1-1)))
						for i = 2; i <= n-1; i++ {
							d.Set(i-1, math.Abs(d.Get(i-1))+math.Abs(e.Get(i-1))+math.Abs(e.Get(i-1-1)))
						}
					}

					//                 Scale D and E so the maximum element is ANORM.
					ix = goblas.Idamax(n, d.Off(0, 1))
					dmax = d.Get(ix - 1)
					goblas.Dscal(n, anorm/dmax, d.Off(0, 1))
					goblas.Dscal(n-1, anorm/dmax, e.Off(0, 1))

				} else if izero > 0 {
					//                 Reuse the last matrix by copying back the zeroed out
					//                 elements.
					if izero == 1 {
						d.Set(0, z.Get(1))
						if n > 1 {
							e.Set(0, z.Get(2))
						}
					} else if izero == n {
						e.Set(n-1-1, z.Get(0))
						d.Set(n-1, z.Get(1))
					} else {
						e.Set(izero-1-1, z.Get(0))
						d.Set(izero-1, z.Get(1))
						e.Set(izero-1, z.Get(2))
					}
				}

				//              For types 8-10, set one row and column of the matrix to
				//              zero.
				izero = 0
				if imat == 8 {
					izero = 1
					z.Set(1, d.Get(0))
					d.Set(0, zero)
					if n > 1 {
						z.Set(2, e.Get(0))
						e.Set(0, zero)
					}
				} else if imat == 9 {
					izero = n
					if n > 1 {
						z.Set(0, e.Get(n-1-1))
						e.Set(n-1-1, zero)
					}
					z.Set(1, d.Get(n-1))
					d.Set(n-1, zero)
				} else if imat == 10 {
					izero = (n + 1) / 2
					if izero > 1 {
						z.Set(0, e.Get(izero-1-1))
						e.Set(izero-1-1, zero)
						z.Set(2, e.Get(izero-1))
						e.Set(izero-1, zero)
					}
					z.Set(1, d.Get(izero-1))
					d.Set(izero-1, zero)
				}
			}

			goblas.Dcopy(n, d.Off(0, 1), d.Off(n, 1))
			if n > 1 {
				goblas.Dcopy(n-1, e.Off(0, 1), e.Off(n, 1))
			}

			//+    TEST 1
			//           Factor A as L*D*L' and compute the ratio
			//              norm(L*D*L' - A) / (n * norm(A) * EPS )
			golapack.Dpttrf(&n, d.Off(n), e.Off(n), &info)

			//           Check error code from DPTTRF.
			if info != izero {
				Alaerh(path, []byte("DPTTRF"), &info, &izero, []byte(" "), &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				goto label100
			}

			if info > 0 {
				rcondc = zero
				goto label90
			}

			Dptt01(&n, d, e, d.Off(n), e.Off(n), work, result.GetPtr(0))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(0) >= (*thresh) {
				if nfail == 0 && nerrs == 0 {
					Alahd(path)
				}
				t.Fail()
				fmt.Printf(" N =%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 1, result.Get(0))
				nfail = nfail + 1
			}
			nrun = nrun + 1

			//           Compute RCONDC = 1 / (norm(A) * norm(inv(A))
			//
			//           Compute norm(A).
			anorm = golapack.Dlanst('1', &n, d, e)

			//           Use DPTTRS to solve for one column at a time of inv(A),
			//           computing the maximum column sum as we go.
			ainvnm = zero
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					x.Set(j-1, zero)
				}
				x.Set(i-1, one)
				golapack.Dpttrs(&n, func() *int { y := 1; return &y }(), d.Off(n), e.Off(n), x.Matrix(lda, opts), &lda, &info)
				ainvnm = math.Max(ainvnm, goblas.Dasum(n, x.Off(0, 1)))
			}
			rcondc = one / math.Max(one, anorm*ainvnm)

			for irhs = 1; irhs <= (*nns); irhs++ {
				nrhs = (*nsval)[irhs-1]

				//           Generate NRHS random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &n, xact.Off(ix-1))
					ix = ix + lda
				}

				//           Set the right hand side.
				Dlaptm(&n, &nrhs, &one, d, e, xact.Matrix(lda, opts), &lda, &zero, b.Matrix(lda, opts), &lda)

				//+    TEST 2
				//           Solve A*x = b and compute the residual.
				golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)
				golapack.Dpttrs(&n, &nrhs, d.Off(n), e.Off(n), x.Matrix(lda, opts), &lda, &info)

				//           Check error code from DPTTRS.
				if info != 0 {
					Alaerh(path, []byte("DPTTRS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
				}

				golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
				Dptt02(&n, &nrhs, d, e, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, result.GetPtr(1))

				//+    TEST 3
				//           Check solution from generated exact solution.
				Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

				//+    TESTS 4, 5, and 6
				//           Use iterative refinement to improve the solution.
				*srnamt = "DPTRFS"
				golapack.Dptrfs(&n, &nrhs, d, e, d.Off(n), e.Off(n), b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), work, &info)

				//           Check error code from DPTRFS.
				if info != 0 {
					Alaerh(path, []byte("DPTRFS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
				}

				Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
				Dptt05(&n, &nrhs, d, e, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), result.Off(4))

				//           Print information about the tests that did not pass the
				//           threshold.
				for k = 2; k <= 6; k++ {
					if result.Get(k-1) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						t.Fail()
						fmt.Printf(" N =%5d, NRHS=%3d, _type %2d, test(%2d) = %12.5f\n", n, nrhs, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
				}
				nrun = nrun + 5
			}

			//+    TEST 7
			//           Estimate the reciprocal of the condition number of the
			//           matrix.
		label90:
			;
			*srnamt = "DPTCON"
			golapack.Dptcon(&n, d.Off(n), e.Off(n), &anorm, &rcond, rwork, &info)

			//           Check error code from DPTCON.
			if info != 0 {
				Alaerh(path, []byte("DPTCON"), &info, func() *int { y := 0; return &y }(), []byte(" "), &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
			}

			result.Set(6, Dget06(&rcond, &rcondc))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(6) >= (*thresh) {
				if nfail == 0 && nerrs == 0 {
					Alahd(path)
				}
				t.Fail()
				fmt.Printf(" N =%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 7, result.Get(6))
				nfail = nfail + 1
			}
			nrun = nrun + 1
		label100:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 953
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
