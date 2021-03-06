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

// ddrvpt tests DPTSV and -SVX.
func ddrvpt(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, a, d, e, b, x, xact, work, rwork *mat.Vector, t *testing.T) {
	var zerot bool
	var dist, fact, _type byte
	var ainvnm, anorm, cond, dmax, one, rcond, rcondc, zero float64
	var i, ia, ifact, imat, in, info, ix, izero, j, k, k1, kl, ku, lda, mode, n, nerrs, nfail, nimat, nrun, nt, ntypes int
	var err error

	result := vf(6)
	z := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	zero = 0.0
	ntypes = 12
	// ntests = 6

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 0, 0, 0, 1

	path := "Dpt"
	alasvmStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrvx(path, t)
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
				goto label110
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
					goto label110
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
					ix = d.Iamax(n, 1)
					dmax = d.Get(ix - 1)
					d.Scal(n, anorm/dmax, 1)
					if n > 1 {
						e.Scal(n-1, anorm/dmax, 1)
					}

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
						z.Set(2, e.Get(izero-1))
						e.Set(izero-1-1, zero)
						e.Set(izero-1, zero)
					}
					z.Set(1, d.Get(izero-1))
					d.Set(izero-1, zero)
				}
			}

			//           Generate NRHS random solution vectors.
			ix = 1
			for j = 1; j <= nrhs; j++ {
				golapack.Dlarnv(2, &iseed, n, xact.Off(ix-1))
				ix = ix + lda
			}

			//           Set the right hand side.
			dlaptm(n, nrhs, one, d, e, xact.Matrix(lda, opts), zero, b.Matrix(lda, opts))

			for ifact = 1; ifact <= 2; ifact++ {
				if ifact == 1 {
					fact = 'F'
				} else {
					fact = 'N'
				}

				//              Compute the condition number for comparison with
				//              the value returned by Dptsvx.
				if zerot {
					if ifact == 1 {
						goto label100
					}
					rcondc = zero

				} else if ifact == 1 {
					//                 Compute the 1-norm of A.
					anorm = golapack.Dlanst('1', n, d, e)

					d.Off(n).Copy(n, d, 1, 1)
					if n > 1 {
						e.Off(n).Copy(n-1, e, 1, 1)
					}

					//                 Factor the matrix A.
					if info, err = golapack.Dpttrf(n, d.Off(n), e.Off(n)); err != nil {
						panic(err)
					}

					//                 Use DPTTRS to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.Set(j-1, zero)
						}
						x.Set(i-1, one)
						if err = golapack.Dpttrs(n, 1, d.Off(n), e.Off(n), x.Matrix(lda, opts)); err != nil {
							panic(err)
						}
						ainvnm = math.Max(ainvnm, x.Asum(n, 1))
					}

					//                 Compute the 1-norm condition number of A.
					if anorm <= zero || ainvnm <= zero {
						rcondc = one
					} else {
						rcondc = (one / anorm) / ainvnm
					}
				}

				if ifact == 2 {
					//                 --- Test DPTSV --
					d.Off(n).Copy(n, d, 1, 1)
					if n > 1 {
						e.Off(n).Copy(n-1, e, 1, 1)
					}
					golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))

					//                 Factor A as L*D*L' and solve the system A*X = B.
					*srnamt = "Dptsv"
					if info, err = golapack.Dptsv(n, nrhs, d.Off(n), e.Off(n), x.Matrix(lda, opts)); err != nil {
						panic(err)
					}

					//                 Check error code from DPTSV .
					if info != izero {
						nerrs = alaerh(path, "Dptsv", info, 0, []byte(" "), n, n, 1, 1, nrhs, imat, nfail, nerrs)
					}
					nt = 0
					if izero == 0 {
						//                    Check the factorization by computing the ratio
						//                       norm(L*D*L' - A) / (n * norm(A) * EPS )
						result.Set(0, dptt01(n, d, e, d.Off(n), e.Off(n), work))

						//                    Compute the residual in the solution.
						golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
						result.Set(1, dptt02(n, nrhs, d, e, x.Matrix(lda, opts), work.Matrix(lda, opts)))

						//                    Check solution from generated exact solution.
						result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
						nt = 3
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							if nfail == 0 && nerrs == 0 {
								aladhd(path)
							}
							t.Fail()
							fmt.Printf(" %s, n=%5d, _type %2d, test %2d, ratio = %12.5f\n", "DPTSV ", n, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt
				}

				//              --- Test Dptsvx ---
				if ifact > 1 {
					//                 Initialize D( N+1:2*N ) and E( N+1:2*N ) to zero.
					for i = 1; i <= n-1; i++ {
						d.Set(n+i-1, zero)
						e.Set(n+i-1, zero)
					}
					if n > 0 {
						d.Set(n+n-1, zero)
					}
				}

				golapack.Dlaset(Full, n, nrhs, zero, zero, x.Matrix(lda, opts))

				//              Solve the system and compute the condition number and
				//              error bounds using Dptsvx.
				*srnamt = "Dptsvx"
				if rcond, info, err = golapack.Dptsvx(fact, n, nrhs, d, e, d.Off(n), e.Off(n), b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work); err != nil {
					panic(err)
				}

				//              Check the error code from Dptsvx.
				if info != izero {
					nerrs = alaerh(path, "Dptsvx", info, 0, []byte{fact}, n, n, 1, 1, nrhs, imat, nfail, nerrs)
				}
				if izero == 0 {
					if ifact == 2 {
						//                    Check the factorization by computing the ratio
						//                       norm(L*D*L' - A) / (n * norm(A) * EPS )
						k1 = 1
						result.Set(0, dptt01(n, d, e, d.Off(n), e.Off(n), work))
					} else {
						k1 = 2
					}

					//                 Compute the residual in the solution.
					golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
					result.Set(1, dptt02(n, nrhs, d, e, x.Matrix(lda, opts), work.Matrix(lda, opts)))

					//                 Check solution from generated exact solution.
					result.Set(2, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

					//                 Check error bounds from iterative refinement.
					dptt05(n, nrhs, d, e, b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
				} else {
					k1 = 6
				}

				//              Check the reciprocal of the condition number.
				result.Set(5, dget06(rcond, rcondc))

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = k1; k <= 6; k++ {
					if result.Get(k-1) >= thresh {
						if nfail == 0 && nerrs == 0 {
							aladhd(path)
						}
						t.Fail()
						fmt.Printf(" %s, fact='%c', n=%5d, _type %2d, test %2d, ratio = %12.5f\n", "Dptsvx", fact, n, imat, k, result.Get(k-1))
						nfail++
					}
				}
				nrun += 7 - k1
			label100:
			}
		label110:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 788
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
