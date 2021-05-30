package lin

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Zdrvgt tests ZGTSV and -SVX.
func Zdrvgt(dotype *[]bool, nn *int, nval *[]int, nrhs *int, thresh *float64, tsterr *bool, a, af, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var zerot bool
	var dist, fact, trans, _type byte
	var ainvnm, anorm, anormi, anormo, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, ifact, imat, in, info, itran, ix, izero, j, k, k1, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrun, nt, ntypes int

	transs := make([]byte, 3)
	result := vf(6)
	z := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 12
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3], transs[0], transs[1], transs[2] = 0, 0, 0, 1, 'N', 'T', 'C'
	path := []byte("ZGT")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrvx(path, t)
	}
	(*infot) = 0

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL.
		n = (*nval)[in-1]
		m = maxint(n-1, 0)
		lda = maxint(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label130
			}

			//           Set up parameters with ZLATB4.
			Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cond, &dist)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Types 1-6:  generate matrices of known condition number.
				koff = maxint(2-ku, 3-maxint(1, n))
				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cond, &anorm, &kl, &ku, 'Z', af.CMatrixOff(koff-1, 3, opts), func() *int { y := 3; return &y }(), work, &info)

				//              Check the error code from ZLATMS.
				if info != 0 {
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
					goto label130
				}
				izero = 0

				if n > 1 {
					goblas.Zcopy(toPtr(n-1), af.Off(3), func() *int { y := 3; return &y }(), a, func() *int { y := 1; return &y }())
					goblas.Zcopy(toPtr(n-1), af.Off(2), func() *int { y := 3; return &y }(), a.Off(n+m+1-1), func() *int { y := 1; return &y }())
				}
				goblas.Zcopy(&n, af.Off(1), func() *int { y := 3; return &y }(), a.Off(m+1-1), func() *int { y := 1; return &y }())
			} else {
				//              Types 7-12:  generate tridiagonal matrices with
				//              unknown condition numbers.
				if !zerot || !(*dotype)[6] {
					//                 Generate a matrix with elements from [-1,1].
					golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(n+2*m), a)
					if anorm != one {
						goblas.Zdscal(toPtr(n+2*m), &anorm, a, func() *int { y := 1; return &y }())
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
				//              the value returned by ZGTSVX.
				if zerot {
					if ifact == 1 {
						goto label120
					}
					rcondo = zero
					rcondi = zero
					//
				} else if ifact == 1 {
					goblas.Zcopy(toPtr(n+2*m), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }())

					//                 Compute the 1-norm and infinity-norm of A.
					anormo = golapack.Zlangt('1', &n, a, a.Off(m+1-1), a.Off(n+m+1-1))
					anormi = golapack.Zlangt('I', &n, a, a.Off(m+1-1), a.Off(n+m+1-1))

					//                 Factor the matrix A.
					golapack.Zgttrf(&n, af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, &info)

					//                 Use ZGTTRS to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.SetRe(j-1, zero)
						}
						x.SetRe(i-1, one)
						golapack.Zgttrs('N', &n, func() *int { y := 1; return &y }(), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, x.CMatrix(lda, opts), &lda, &info)
						ainvnm = maxf64(ainvnm, goblas.Dzasum(&n, x, func() *int { y := 1; return &y }()))
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
						golapack.Zgttrs('C', &n, func() *int { y := 1; return &y }(), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, x.CMatrix(lda, opts), &lda, &info)
						ainvnm = maxf64(ainvnm, goblas.Dzasum(&n, x, func() *int { y := 1; return &y }()))
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
						golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &n, xact.Off(ix-1))
						ix = ix + lda
					}

					//                 Set the right hand side.
					golapack.Zlagtm(trans, &n, nrhs, &one, a, a.Off(m+1-1), a.Off(n+m+1-1), xact.CMatrix(lda, opts), &lda, &zero, b.CMatrix(lda, opts), &lda)

					if ifact == 2 && itran == 1 {
						//                    --- Test ZGTSV  ---
						//
						//                    Solve the system using Gaussian elimination with
						//                    partial pivoting.
						goblas.Zcopy(toPtr(n+2*m), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }())
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)

						*srnamt = "ZGTSV "
						golapack.Zgtsv(&n, nrhs, af, af.Off(m+1-1), af.Off(n+m+1-1), x.CMatrix(lda, opts), &lda, &info)

						//                    Check error code from ZGTSV .
						if info != izero {
							Alaerh(path, []byte("ZGTSV "), &info, &izero, []byte{' '}, &n, &n, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), nrhs, &imat, &nfail, &nerrs)
						}
						nt = 1
						if izero == 0 {
							//                       Check residual of computed solution.
							golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
							Zgtt02(trans, &n, nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, result.GetPtr(1))

							//                       Check solution from generated exact solution.
							Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))
							nt = 3
						}

						//                    Print information about the tests that did not pass
						//                    the threshold.
						for k = 2; k <= nt; k++ {
							if result.Get(k-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Aladhd(path)
								}
								fmt.Printf(" %s, N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "ZGTSV ", n, imat, k, result.Get(k-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt - 1
					}

					//                 --- Test ZGTSVX ---
					if ifact > 1 {
						//                    Initialize AF to zero.
						for i = 1; i <= 3*n-2; i++ {
							af.SetRe(i-1, zero)
						}
					}
					golapack.Zlaset('F', &n, nrhs, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), x.CMatrix(lda, opts), &lda)

					//                 Solve the system and compute the condition number and
					//                 error bounds using ZGTSVX.
					*srnamt = "ZGTSVX"
					golapack.Zgtsvx(fact, trans, &n, nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, &rcond, rwork, rwork.Off((*nrhs)+1-1), work, rwork.Off(2*(*nrhs)+1-1), &info)

					//                 Check the error code from ZGTSVX.
					if info != izero {
						Alaerh(path, []byte("ZGTSVX"), &info, &izero, []byte{fact, trans}, &n, &n, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), nrhs, &imat, &nfail, &nerrs)
					}

					if ifact >= 2 {
						//                    Reconstruct matrix from factors and compute
						//                    residual.
						Zgtt01(&n, a, a.Off(m+1-1), a.Off(n+m+1-1), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))
						k1 = 1
					} else {
						k1 = 2
					}

					if info == 0 {
						// trfcon = false

						//                    Check residual of computed solution.
						golapack.Zlacpy('F', &n, nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
						Zgtt02(trans, &n, nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, result.GetPtr(1))

						//                    Check solution from generated exact solution.
						Zget04(&n, nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

						//                    Check the error bounds from iterative refinement.
						Zgtt05(trans, &n, nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off((*nrhs)+1-1), result.Off(3))
						nt = 5
					}

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = k1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Aladhd(path)
							}
							fmt.Printf(" %s, FACT='%c', TRANS='%c', N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "ZGTSVX", fact, trans, n, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}

					//                 Check the reciprocal of the condition number.
					result.Set(5, Dget06(&rcond, &rcondc))
					if result.Get(5) >= (*thresh) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Aladhd(path)
						}
						fmt.Printf(" %s, FACT='%c', TRANS='%c', N =%5d, _type %2d, test %2d, ratio = %12.5f\n", "ZGTSVX", fact, trans, n, imat, k, result.Get(k-1))
						nfail = nfail + 1
					}
					nrun = nrun + nt - k1 + 2

				}
			label120:
			}
		label130:
		}
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
