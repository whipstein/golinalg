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

// dchkpt tests DPTTRF, -TRS, -RFS, and -CON
func dchkpt(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, a, d, e, b, x, xact, work, rwork *mat.Vector, t *testing.T) {
	var zerot bool
	var dist, _type byte
	var ainvnm, anorm, cond, dmax, one, rcond, rcondc, zero float64
	var i, ia, imat, in, info, irhs, ix, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int
	var err error

	result := vf(7)
	z := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 12

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 0, 0, 0, 1

	path := "Dpt"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrgt(path, t)
	}
	(*infot) = 0

	for in = 1; in <= nn; in++ {
		//        Do for each value of N in NVAL.
		n = nval[in-1]
		lda = max(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if n > 0 && !dotype[imat-1] {
				goto label100
			}

			//           Set up parameters with DLATB4.
			_type, kl, ku, anorm, mode, cond, dist = dlatb4(path, imat, n, n)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Type 1-6:  generate a symmetric tridiagonal matrix of
				//              known condition number in lower triangular band storage.
				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cond, anorm, kl, ku, 'B', a.Matrix(2, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte(" "), n, n, kl, ku, -1, imat, nfail, nerrs)
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
				if !zerot || !dotype[6] {
					//                 Let D and E have values from [-1,1].
					golapack.Dlarnv(2, &iseed, n, d)
					golapack.Dlarnv(2, &iseed, n-1, e)

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
			if info, err = golapack.Dpttrf(n, d.Off(n), e.Off(n)); err != nil {
				panic(err)
			}

			//           Check error code from DPTTRF.
			if info != izero {
				nerrs = alaerh(path, "Dpttrf", info, 0, []byte(" "), n, n, -1, -1, -1, imat, nfail, nerrs)
				goto label100
			}

			if info > 0 {
				rcondc = zero
				goto label90
			}

			result.Set(0, dptt01(n, d, e, d.Off(n), e.Off(n), work))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(0) >= thresh {
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				t.Fail()
				fmt.Printf(" n=%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 1, result.Get(0))
				nfail++
			}
			nrun++

			//           Compute RCONDC = 1 / (norm(A) * norm(inv(A))
			//
			//           Compute norm(A).
			anorm = golapack.Dlanst('1', n, d, e)

			//           Use DPTTRS to solve for one column at a time of inv(A),
			//           computing the maximum column sum as we go.
			ainvnm = zero
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					x.Set(j-1, zero)
				}
				x.Set(i-1, one)
				if err = golapack.Dpttrs(n, 1, d.Off(n), e.Off(n), x.Matrix(lda, opts)); err != nil {
					panic(err)
				}
				ainvnm = math.Max(ainvnm, goblas.Dasum(n, x.Off(0, 1)))
			}
			rcondc = one / math.Max(one, anorm*ainvnm)

			for irhs = 1; irhs <= nns; irhs++ {
				nrhs = nsval[irhs-1]

				//           Generate nrhs random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Dlarnv(2, &iseed, n, xact.Off(ix-1))
					ix = ix + lda
				}

				//           Set the right hand side.
				dlaptm(n, nrhs, one, d, e, xact.Matrix(lda, opts), zero, b.Matrix(lda, opts))

				//+    TEST 2
				//           Solve A*x = b and compute the residual.
				golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))
				if err = golapack.Dpttrs(n, nrhs, d.Off(n), e.Off(n), x.Matrix(lda, opts)); err != nil {
					nerrs = alaerh(path, "Dpttrs", info, 0, []byte(" "), n, n, -1, -1, nrhs, imat, nfail, nerrs)
				}

				golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
				result.Set(1, dptt02(n, nrhs, d, e, x.Matrix(lda, opts), work.Matrix(lda, opts)))

				//+    TEST 3
				//           Check solution from generated exact solution.
				result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

				//+    TESTS 4, 5, and 6
				//           Use iterative refinement to improve the solution.
				*srnamt = "Dptrfs"
				if err = golapack.Dptrfs(n, nrhs, d, e, d.Off(n), e.Off(n), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work); err != nil {
					nerrs = alaerh(path, "Dptrfs", info, 0, []byte(" "), n, n, -1, -1, nrhs, imat, nfail, nerrs)
				}

				result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
				dptt05(n, nrhs, d, e, b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

				//           Print information about the tests that did not pass the
				//           threshold.
				for k = 2; k <= 6; k++ {
					if result.Get(k-1) >= thresh {
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						t.Fail()
						fmt.Printf(" n=%5d, nrhs=%3d, _type %2d, test(%2d) = %12.5f\n", n, nrhs, imat, k, result.Get(k-1))
						nfail++
					}
				}
				nrun += 5
			}

			//+    TEST 7
			//           Estimate the reciprocal of the condition number of the
			//           matrix.
		label90:
			;
			*srnamt = "Dptcon"
			if rcond, err = golapack.Dptcon(n, d.Off(n), e.Off(n), anorm, rwork); err != nil {
				nerrs = alaerh(path, "Dptcon", info, 0, []byte(" "), n, n, -1, -1, -1, imat, nfail, nerrs)
			}

			result.Set(6, dget06(rcond, rcondc))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(6) >= thresh {
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				t.Fail()
				fmt.Printf(" n=%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 7, result.Get(6))
				nfail++
			}
			nrun++
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
	alasum(path, nfail, nrun, nerrs)
}
