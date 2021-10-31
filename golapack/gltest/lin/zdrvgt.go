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

// zdrvgt tests Zgtsvand -SVX.
func zdrvgt(dotype []bool, nn int, nval []int, nrhs int, thresh float64, tsterr bool, a, af, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var zerot bool
	var dist, fact, _type byte
	var trans mat.MatTrans
	var ainvnm, anorm, anormi, anormo, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, ifact, imat, in, info, ix, izero, j, k, k1, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrun, nt, ntypes int
	var err error

	result := vf(6)
	z := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 12
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 0, 0, 0, 1
	path := "Zgt"
	alasvmStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrvx(path, t)
	}
	(*infot) = 0

	for in = 1; in <= nn; in++ {
		//        Do for each value of N in NVAL.
		n = nval[in-1]
		m = max(n-1, 0)
		lda = max(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label130
			}

			//           Set up parameters with ZLATB4.
			_type, kl, ku, anorm, mode, cond, dist = zlatb4(path, imat, n, n)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Types 1-6:  generate matrices of known condition number.
				koff = max(2-ku, 3-max(1, n))
				*srnamt = "Zlatms"
				if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cond, anorm, kl, ku, 'Z', af.CMatrixOff(koff-1, 3, opts), work); err != nil {
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, n, n, kl, ku, -1, imat, nfail, nerrs)
					goto label130
				}
				izero = 0

				if n > 1 {
					goblas.Zcopy(n-1, af.Off(3, 3), a.Off(0, 1))
					goblas.Zcopy(n-1, af.Off(2, 3), a.Off(n+m, 1))
				}
				goblas.Zcopy(n, af.Off(1, 3), a.Off(m, 1))
			} else {
				//              Types 7-12:  generate tridiagonal matrices with
				//              unknown condition numbers.
				if !zerot || !dotype[6] {
					//                 Generate a matrix with elements from [-1,1].
					golapack.Zlarnv(2, &iseed, n+2*m, a)
					if anorm != one {
						goblas.Zdscal(n+2*m, anorm, a.Off(0, 1))
					}
				} else if izero > 0 {
					//                 Reuse the last matrix by copying back the zeroed out
					//                 elements.
					if izero == 1 {
						a.SetRe(n-1, z.Get(1))
						if n > 1 {
							a.SetRe(0, z.Get(2))
						}
					} else if izero == n {
						a.SetRe(3*n-2-1, z.Get(0))
						a.SetRe(2*n-1-1, z.Get(1))
					} else {
						a.SetRe(2*n-2+izero-1, z.Get(0))
						a.SetRe(n-1+izero-1, z.Get(1))
						a.SetRe(izero-1, z.Get(2))
					}
				}

				//              If IMAT > 7, set one column of the matrix to 0.
				if !zerot {
					izero = 0
				} else if imat == 8 {
					izero = 1
					a.SetRe(n-1, zero)
					if n > 1 {
						z.Set(2, real(a.Get(0)))
						a.SetRe(0, zero)
					}
				} else if imat == 9 {
					izero = n
					z.Set(0, real(a.Get(3*n-2-1)))
					z.Set(1, real(a.Get(2*n-1-1)))
					a.SetRe(3*n-2-1, zero)
					a.SetRe(2*n-1-1, zero)
				} else {
					izero = (n + 1) / 2
					for i = izero; i <= n-1; i++ {
						a.SetRe(2*n-2+i-1, zero)
						a.SetRe(n-1+i-1, zero)
						a.SetRe(i-1, zero)
					}
					a.SetRe(3*n-2-1, zero)
					a.SetRe(2*n-1-1, zero)
				}
			}

			for ifact = 1; ifact <= 2; ifact++ {
				if ifact == 1 {
					fact = 'F'
				} else {
					fact = 'N'
				}

				//              Compute the condition number for comparison with
				//              the value returned by Zgtsvx.
				if zerot {
					if ifact == 1 {
						goto label120
					}
					rcondo = zero
					rcondi = zero
					//
				} else if ifact == 1 {
					goblas.Zcopy(n+2*m, a.Off(0, 1), af.Off(0, 1))

					//                 Compute the 1-norm and infinity-norm of A.
					anormo = golapack.Zlangt('1', n, a, a.Off(m), a.Off(n+m))
					anormi = golapack.Zlangt('I', n, a, a.Off(m), a.Off(n+m))

					//                 Factor the matrix A.
					if info, err = golapack.Zgttrf(n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork); err != nil {
						panic(err)
					}

					//                 Use ZGTTRS to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.SetRe(j-1, zero)
						}
						x.SetRe(i-1, one)
						if err = golapack.Zgttrs(NoTrans, n, 1, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, x.CMatrix(lda, opts)); err != nil {
							panic(err)
						}
						ainvnm = math.Max(ainvnm, goblas.Dzasum(n, x.Off(0, 1)))
					}

					//                 Compute the 1-norm condition number of A.
					if anormo <= zero || ainvnm <= zero {
						rcondo = one
					} else {
						rcondo = (one / anormo) / ainvnm
					}

					//                 Use ZGTTRS to solve for one column at a time of
					//                 inv(A'), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.SetRe(j-1, zero)
						}
						x.SetRe(i-1, one)
						if err = golapack.Zgttrs(ConjTrans, n, 1, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, x.CMatrix(lda, opts)); err != nil {
							panic(err)
						}
						ainvnm = math.Max(ainvnm, goblas.Dzasum(n, x.Off(0, 1)))
					}

					//                 Compute the infinity-norm condition number of A.
					if anormi <= zero || ainvnm <= zero {
						rcondi = one
					} else {
						rcondi = (one / anormi) / ainvnm
					}
				}

				for _, trans = range mat.IterMatTrans() {
					if trans == NoTrans {
						rcondc = rcondo
					} else {
						rcondc = rcondi
					}

					//                 Generate nrhs random solution vectors.
					ix = 1
					for j = 1; j <= nrhs; j++ {
						golapack.Zlarnv(2, &iseed, n, xact.Off(ix-1))
						ix = ix + lda
					}

					//                 Set the right hand side.
					golapack.Zlagtm(trans, n, nrhs, one, a, a.Off(m), a.Off(n+m), xact.CMatrix(lda, opts), zero, b.CMatrix(lda, opts))

					if ifact == 2 && trans == NoTrans {
						//                    --- Test Zgtsv ---
						//
						//                    Solve the system using Gaussian elimination with
						//                    partial pivoting.
						goblas.Zcopy(n+2*m, a.Off(0, 1), af.Off(0, 1))
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

						*srnamt = "Zgtsv"
						if info, err = golapack.Zgtsv(n, nrhs, af, af.Off(m), af.Off(n+m), x.CMatrix(lda, opts)); err != nil || info != izero {
							nerrs = alaerh(path, "Zgtsv", info, 0, []byte{' '}, n, n, 1, 1, nrhs, imat, nfail, nerrs)
						}
						nt = 1
						if izero == 0 {
							//                       Check residual of computed solution.
							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
							*result.GetPtr(1) = zgtt02(trans, n, nrhs, a, a.Off(m), a.Off(n+m), x.CMatrix(lda, opts), work.CMatrix(lda, opts))

							//                       Check solution from generated exact solution.
							*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							nt = 3
						}

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= nt; k++ {
							if result.Get(k-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									aladhd(path)
								}
								fmt.Printf(" %s, n=%5d, _type %2d, test %2d, ratio = %12.5f\n", "Zgtsv", n, imat, k, result.Get(k-1))
								nfail++
							}
						}
						nrun = nrun + nt - 1
					}

					//                 --- Test Zgtsvx ---
					if ifact > 1 {
						//                    Initialize AF to zero.
						for i = 1; i <= 3*n-2; i++ {
							af.SetRe(i-1, zero)
						}
					}
					golapack.Zlaset(Full, n, nrhs, complex(zero, 0), complex(zero, 0), x.CMatrix(lda, opts))

					//                 Solve the system and compute the condition number and
					//                 error bounds using Zgtsvx.
					*srnamt = "Zgtsvx"
					if rcond, info, err = golapack.Zgtsvx(fact, trans, n, nrhs, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil || info != izero {
						nerrs = alaerh(path, "Zgtsvx", info, 0, []byte{fact, trans.Byte()}, n, n, 1, 1, nrhs, imat, nfail, nerrs)
					}

					if ifact >= 2 {
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						*result.GetPtr(0) = zgtt01(n, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, work.CMatrix(lda, opts), rwork)
						k1 = 1
					} else {
						k1 = 2
					}

					if info == 0 {
						// trfcon = false

						//                    Check residual of computed solution.
						golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
						*result.GetPtr(1) = zgtt02(trans, n, nrhs, a, a.Off(m), a.Off(n+m), x.CMatrix(lda, opts), work.CMatrix(lda, opts))

						//                    Check solution from generated exact solution.
						*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

						//                    Check the error bounds from iterative refinement.
						zgtt05(trans, n, nrhs, a, a.Off(m), a.Off(n+m), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(3))
						nt = 5
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								aladhd(path)
							}
							fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test %2d, ratio = %12.5f\n", "Zgtsvx", fact, trans, n, imat, k, result.Get(k-1))
							nfail++
						}
					}

					//                 Check the reciprocal of the condition number.
					result.Set(5, dget06(rcond, rcondc))
					if result.Get(5) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							aladhd(path)
						}
						fmt.Printf(" %s, fact='%c', trans=%s, n=%5d, _type %2d, test %2d, ratio = %12.5f\n", "Zgtsvx", fact, trans, n, imat, k, result.Get(k-1))
						nfail++
					}
					nrun = nrun + nt - k1 + 2

				}
			label120:
			}
		label130:
		}
	}

	//     Print a summary of the results.
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
