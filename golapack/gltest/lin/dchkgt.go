package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkgt tests DGTTRF, -TRS, -RFS, and -CON
func dchkgt(dotype []bool, nval, nsval []int, thresh float64, tsterr bool, a, af, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, _type byte
	var trans mat.MatTrans
	var ainvnm, anorm, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, imat, info, ix, izero, j, k, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int
	var err error
	var _iwork []int

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

	path := "Dgt"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrge(path, t)
	}
	(*infot) = 0

	for _, n = range nval {
		//        Do for each value of N in NVAL.
		m = max(n-1, 0)
		lda = max(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				continue
			}

			//           Set up parameters with DLATB4.
			_type, kl, ku, anorm, mode, cond, dist = dlatb4(path, imat, n, n)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Types 1-6:  generate matrices of known condition number.
				koff = max(2-ku, 3-max(1, n))
				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(n, n, dist, &iseed, _type, rwork, mode, cond, anorm, kl, ku, 'Z', af.Off(koff-1).Matrix(3, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte(" "), n, n, kl, ku, -1, imat, nfail, nerrs)
					continue
				}
				izero = 0

				if n > 1 {
					a.Copy(n-1, af.Off(3), 3, 1)
					a.Off(n+m).Copy(n-1, af.Off(2), 3, 1)
				}
				a.Off(m).Copy(n, af.Off(1), 3, 1)
			} else {
				//              Types 7-12:  generate tridiagonal matrices with
				//              unknown condition numbers.
				if !zerot || !dotype[6] {
					//                 Generate a matrix with elements from [-1,1].
					golapack.Dlarnv(2, &iseed, n+2*m, a)
					if anorm != one {
						a.Scal(n+2*m, anorm, 1)
					}
				} else if izero > 0 {
					//                 Reuse the last matrix by copying back the zeroed out
					//                 elements.
					if izero == 1 {
						a.Set(n-1, z.Get(1))
						if n > 1 {
							a.Set(0, z.Get(2))
						}
					} else if izero == n {
						a.Set(3*n-2-1, z.Get(0))
						a.Set(2*n-1-1, z.Get(1))
					} else {
						a.Set(2*n-2+izero-1, z.Get(0))
						a.Set(n-1+izero-1, z.Get(1))
						a.Set(izero-1, z.Get(2))
					}
				}

				//              If IMAT > 7, set one column of the matrix to 0.
				if !zerot {
					izero = 0
				} else if imat == 8 {
					izero = 1
					z.Set(1, a.Get(n-1))
					a.Set(n-1, zero)
					if n > 1 {
						z.Set(2, a.Get(0))
						a.Set(0, zero)
					}
				} else if imat == 9 {
					izero = n
					z.Set(0, a.Get(3*n-2-1))
					z.Set(1, a.Get(2*n-1-1))
					a.Set(3*n-2-1, zero)
					a.Set(2*n-1-1, zero)
				} else {
					izero = (n + 1) / 2
					for i = izero; i <= n-1; i++ {
						a.Set(2*n-2+i-1, zero)
						a.Set(n-1+i-1, zero)
						a.Set(i-1, zero)
					}
					a.Set(3*n-2-1, zero)
					a.Set(2*n-1-1, zero)
				}
			}

			//+    TEST 1
			//           Factor A as L*U and compute the ratio
			//              norm(L*U - A) / (n * norm(A) * EPS )
			af.Copy(n+2*m, a, 1, 1)
			*srnamt = "Dgttrf"
			if info, err = golapack.Dgttrf(n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork); err != nil {
				panic(err)
			}

			//           Check error code from DGTTRF.
			if info != izero {
				nerrs = alaerh(path, "Dgttrf", info, 0, []byte(" "), n, n, 1, 1, -1, imat, nfail, nerrs)
			}
			trfcon = info != 0

			result.Set(0, dgtt01(n, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, work.Matrix(lda, opts), rwork))

			//           Print the test ratio if it is .GE. THRESH.
			if result.Get(0) >= thresh {
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				t.Fail()
				fmt.Printf("            N =%5d,           _type %2d, test(%2d) = %12.5f\n", n, imat, 1, result.Get(0))
				nfail++
			}
			nrun++

			for _, trans = range mat.IterMatTrans(false) {
				if trans == NoTrans {
					norm = 'O'
				} else {
					norm = 'I'
				}
				anorm = golapack.Dlangt(norm, n, a, a.Off(m), a.Off(n+m))

				if !trfcon {
					//                 Use DGTTRS to solve for one column at a time of inv(A)
					//                 or inv(A^T), computing the maximum column sum as we
					//                 go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.Set(j-1, zero)
						}
						x.Set(i-1, one)
						if err = golapack.Dgttrs(trans, n, 1, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, x.Matrix(lda, opts)); err != nil {
							panic(err)
						}
						ainvnm = math.Max(ainvnm, x.Asum(n, 1))
					}

					//                 Compute RCONDC = 1 / (norm(A) * norm(inv(A))
					if anorm <= zero || ainvnm <= zero {
						rcondc = one
					} else {
						rcondc = (one / anorm) / ainvnm
					}
					if trans == NoTrans {
						rcondo = rcondc
					} else {
						rcondi = rcondc
					}
				} else {
					rcondc = zero
				}

				//+    TEST 7
				//              Estimate the reciprocal of the condition number of the
				//              matrix.
				*srnamt = "Dgtcon"
				if rcond, err = golapack.Dgtcon(norm, n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, anorm, work, toSlice(&iwork, n)); err != nil {
					nerrs = alaerh(path, "Dgtcon", info, 0, []byte{norm}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				result.Set(6, dget06(rcond, rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(6) >= thresh {
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					t.Fail()
					fmt.Printf(" NORM ='%c', N =%5d,           _type %2d, test(%2d) = %12.5f\n", norm, n, imat, 7, result.Get(6))
					nfail++
				}
				nrun++
			}
			for i, val := range _iwork {
				iwork[n+i] = val
			}

			//           Skip the remaining tests if the matrix is singular.
			if trfcon {
				goto label100
			}

			for _, nrhs = range nsval {

				//              Generate NRHS random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Dlarnv(2, &iseed, n, xact.Off(ix-1))
					ix = ix + lda
				}

				for _, trans = range mat.IterMatTrans() {
					if trans == NoTrans {
						rcondc = rcondo
					} else {
						rcondc = rcondi
					}

					//                 Set the right hand side.
					golapack.Dlagtm(trans, n, nrhs, one, a, a.Off(m), a.Off(n+m), xact.Matrix(lda, opts), zero, b.Matrix(lda, opts))

					//+    TEST 2
					//                 Solve op(A) * X = B and compute the residual.
					golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))
					*srnamt = "Dgttrs"
					if err = golapack.Dgttrs(trans, n, nrhs, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, x.Matrix(lda, opts)); err != nil {
						panic(err)
					}

					//                 Check error code from DGTTRS.
					if info != 0 {
						nerrs = alaerh(path, "Dgttrs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
					result.Set(1, dgtt02(trans, n, nrhs, a, a.Off(m), a.Off(n+m), x.Matrix(lda, opts), work.Matrix(lda, opts)))

					//+    TEST 3
					//                 Check solution from generated exact solution.
					result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

					//+    TESTS 4, 5, and 6
					//                 Use iterative refinement to improve the solution.
					*srnamt = "Dgtrfs"
					if err = golapack.Dgtrfs(trans, n, nrhs, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, toSlice(&iwork, n)); err != nil {
						nerrs = alaerh(path, "Dgtrfs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}
					for i, val := range _iwork {
						iwork[n+i] = val
					}

					result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
					dgtt05(trans, n, nrhs, a, a.Off(m), a.Off(n+m), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 2; k <= 6; k++ {
						if result.Get(k-1) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							t.Fail()
							fmt.Printf(" TRANS=%s, N =%5d, NRHS=%3d, _type %2d, test(%2d) = %12.5f\n", trans, n, nrhs, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 5
				}
			}

		label100:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 2694
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
