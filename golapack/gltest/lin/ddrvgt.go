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

// Ddrvgt tests DGTSV and -SVX.
func Ddrvgt(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, a, af, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, fact, trans, _type byte
	var ainvnm, anorm, anormi, anormo, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, ifact, imat, in, info, itran, ix, izero, j, k, k1, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrun, nt, ntypes int

	transs := make([]byte, 3)
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

	iseedy[0], iseedy[1], iseedy[2], iseedy[3], transs[0], transs[1], transs[2] = 0, 0, 0, 1, 'N', 'T', 'C'

	path := []byte("DGT")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrvx(path, t)
	}
	(*infot) = 0

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL.
		n = (*nval)[in-1]
		m = max(n-1, 0)
		lda = max(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label130
			}

			//           Set up parameters with DLATB4.
			Dlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cond, &dist)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Types 1-6:  generate matrices of known condition number.
				koff = max(2-ku, 3-max(1, n))
				*srnamt = "DLATMS"
				matgen.Dlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cond, &anorm, &kl, &ku, 'Z', af.MatrixOff(koff-1, 3, opts), toPtr(3), work, &info)

				//              Check the error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, toPtr(0), []byte(" "), &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
					goto label130
				}
				izero = 0
				//
				if n > 1 {
					goblas.Dcopy(n-1, af.Off(3, 3), a.Off(0, 1))
					goblas.Dcopy(n-1, af.Off(2, 3), a.Off(n+m, 1))
				}
				goblas.Dcopy(n, af.Off(1, 3), a.Off(m, 1))
			} else {
				//              Types 7-12:  generate tridiagonal matrices with
				//              unknown condition numbers.
				if !zerot || !(*dotype)[6] {
					//                 Generate a matrix with elements from [-1,1].
					golapack.Dlarnv(toPtr(2), &iseed, toPtr(n+2*m), a)
					if anorm != one {
						goblas.Dscal(n+2*m, anorm, a.Off(0, 1))
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

			for ifact = 1; ifact <= 2; ifact++ {
				if ifact == 1 {
					fact = 'F'
				} else {
					fact = 'N'
				}

				//              Compute the condition number for comparison with
				//              the value returned by DGTSVX.
				if zerot {
					if ifact == 1 {
						goto label120
					}
					rcondo = zero
					rcondi = zero

				} else if ifact == 1 {
					goblas.Dcopy(n+2*m, a.Off(0, 1), af.Off(0, 1))

					//                 Compute the 1-norm and infinity-norm of A.
					anormo = golapack.Dlangt('1', &n, a, a.Off(m), a.Off(n+m))
					anormi = golapack.Dlangt('I', &n, a, a.Off(m), a.Off(n+m))

					//                 Factor the matrix A.
					golapack.Dgttrf(&n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, &info)

					//                 Use DGTTRS to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.Set(j-1, zero)
						}
						x.Set(i-1, one)
						golapack.Dgttrs('N', &n, toPtr(1), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, x.Matrix(lda, opts), &lda, &info)
						ainvnm = math.Max(ainvnm, goblas.Dasum(n, x.Off(0, 1)))
					}

					//                 Compute the 1-norm condition number of A.
					if anormo <= zero || ainvnm <= zero {
						rcondo = one
					} else {
						rcondo = (one / anormo) / ainvnm
					}

					//                 Use DGTTRS to solve for one column at a time of
					//                 inv(A'), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.Set(j-1, zero)
						}
						x.Set(i-1, one)
						golapack.Dgttrs('T', &n, toPtr(1), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, x.Matrix(lda, opts), &lda, &info)
						ainvnm = math.Max(ainvnm, goblas.Dasum(n, x.Off(0, 1)))
					}

					//                 Compute the infinity-norm condition number of A.
					if anormi <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anormi) / ainvnm
					}
				}

				for itran = 1; itran <= 3; itran++ {
					trans = transs[itran-1]
					if itran == 1 {
						rcondc = rcondo
					} else {
						rcondc = rcondi
					}

					//                 Generate NRHS random solution vectors.
					ix = 1
					for j = 1; j <= (*nrhs); j++ {
						golapack.Dlarnv(toPtr(2), &iseed, &n, xact.Off(ix-1))
						ix = ix + lda
					}

					//                 Set the right hand side.
					golapack.Dlagtm(trans, &n, nrhs, &one, a, a.Off(m), a.Off(n+m), xact.Matrix(lda, opts), &lda, &zero, b.Matrix(lda, opts), &lda)

					if ifact == 2 && itran == 1 {
						//                    --- Test DGTSV  ---
						//
						//                    Solve the system using Gaussian elimination with
						//                    partial pivoting.
						goblas.Dcopy(n+2*m, a.Off(0, 1), af.Off(0, 1))
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)

						*srnamt = "DGTSV "
						golapack.Dgtsv(&n, nrhs, af, af.Off(m), af.Off(n+m), x.Matrix(lda, opts), &lda, &info)

						//                    Check error code from DGTSV .
						if info != izero {
							Alaerh(path, []byte("DGTSV "), &info, &izero, []byte(" "), &n, &n, toPtr(1), toPtr(1), nrhs, &imat, &nfail, &nerrs)
						}
						nt = 1
						if izero == 0 {
							//                       Check residual of computed solution.
							golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dgtt02(trans, &n, nrhs, a, a.Off(m), a.Off(n+m), x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, result.GetPtr(1))

							//                       Check solution from generated exact solution.
							Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							nt = 3
						}

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= nt; k++ {
							if result.Get(k-1) >= (*thresh) {
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								t.Fail()
								fmt.Printf(" %s, N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "DGTSV ", n, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt - 1
					}

					//                 --- Test DGTSVX ---
					if ifact > 1 {
						//                    Initialize AF to zero.
						for i = 1; i <= 3*n-2; i++ {
							af.Set(i-1, zero)
						}
					}
					golapack.Dlaset('F', &n, nrhs, &zero, &zero, x.Matrix(lda, opts), &lda)

					//                 Solve the system and compute the condition number and
					//                 error bounds using DGTSVX.
					*srnamt = "DGTSVX"
					golapack.Dgtsvx(fact, trans, &n, nrhs, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)), work, toSlice(iwork, n), &info)

					//                 Check the error code from DGTSVX.
					if info != izero {
						Alaerh(path, []byte("DGTSVX"), &info, &izero, []byte{fact, trans}, &n, &n, toPtr(1), toPtr(1), nrhs, &imat, &nfail, &nerrs)
					}

					if ifact >= 2 {
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						Dgtt01(&n, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), iwork, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(0))
						k1 = 1
					} else {
						k1 = 2
					}

					if info == 0 {
						//                    Check residual of computed solution.
						golapack.Dlacpy('F', &n, nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
						Dgtt02(trans, &n, nrhs, a, a.Off(m), a.Off(n+m), x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Dget04(&n, nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

						//                    Check the error bounds from iterative refinement.
						Dgtt05(trans, &n, nrhs, a, a.Off(m), a.Off(n+m), b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)), result.Off(3))
						nt = 5
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Aladhd(path)
							}
							t.Fail()
							fmt.Printf(" %s, FACT='%c', TRANS='%c', N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "DGTSVX", fact, trans, n, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}

					//                 Check the reciprocal of the condition number.
					result.Set(5, Dget06(&rcond, &rcondc))
					if result.Get(5) >= (*thresh) {
						if nfail == 0 && nerrs == 0 {
							Aladhd(path)
						}
						t.Fail()
						fmt.Printf(" %s, FACT='%c', TRANS='%c', N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "DGTSVX", fact, trans, n, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
					nrun = nrun + nt - k1 + 2

				}
			label120:
			}
		label130:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 2033
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
