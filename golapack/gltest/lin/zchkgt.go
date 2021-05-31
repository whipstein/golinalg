package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkgt tests ZGTTRF, -TRS, -RFS, and -CON
func Zchkgt(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, a, af, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, trans, _type byte
	var ainvnm, anorm, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, imat, in, info, irhs, itran, ix, izero, j, k, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int

	transs := make([]byte, 3)
	z := cvf(3)
	result := vf(7)
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
		Zerrge(path, t)
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
				goto label100
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
					goto label100
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
					//                 Generate a matrix with elements whose real and
					//                 imaginary parts are from [-1,1].
					golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(n+2*m), a)
					if anorm != one {
						goblas.Zdscal(toPtr(n+2*m), &anorm, a, func() *int { y := 1; return &y }())
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
					a.SetRe(n-1, zero)
					if n > 1 {
						z.Set(2, a.Get(0))
						a.SetRe(0, zero)
					}
				} else if imat == 9 {
					izero = n
					z.Set(0, a.Get(3*n-2-1))
					z.Set(1, a.Get(2*n-1-1))
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

			//+    TEST 1
			//           Factor A as L*U and compute the ratio
			//              norm(L*U - A) / (n * norm(A) * EPS )
			goblas.Zcopy(toPtr(n+2*m), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }())
			*srnamt = "ZGTTRF"
			golapack.Zgttrf(&n, af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, &info)

			//           Check error code from ZGTTRF.
			if info != izero {
				Alaerh(path, []byte("ZGTTRF"), &info, &izero, []byte{' '}, &n, &n, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), &imat, &nfail, &nerrs)
			}
			trfcon = info != 0

			Zgtt01(&n, a, a.Off(m+1-1), a.Off(n+m+1-1), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, work.CMatrix(lda, opts), &lda, rwork, result.GetPtr(0))

			//           Print the test ratio if it is .GE. THRESH.
			if result.Get(0) >= (*thresh) {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					Alahd(path)
				}
				fmt.Printf("            N =%5d,           _type %2d, test(%2d) = %12.5f\n", n, imat, 1, result.Get(0))
				nfail = nfail + 1
			}
			nrun = nrun + 1

			for itran = 1; itran <= 2; itran++ {
				trans = transs[itran-1]
				if itran == 1 {
					norm = 'O'
				} else {
					norm = 'I'
				}
				anorm = golapack.Zlangt(norm, &n, a, a.Off(m+1-1), a.Off(n+m+1-1))

				if !trfcon {
					//                 Use ZGTTRS to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.SetRe(j-1, zero)
						}
						x.SetRe(i-1, one)
						golapack.Zgttrs(trans, &n, func() *int { y := 1; return &y }(), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, x.CMatrix(lda, opts), &lda, &info)
						ainvnm = maxf64(ainvnm, goblas.Dzasum(&n, x, func() *int { y := 1; return &y }()))
					}

					//                 Compute RCONDC = 1 / (norm(A) * norm(inv(A))
					if anorm <= zero || ainvnm <= zero {
						rcondc = one
					} else {
						rcondc = (one / anorm) / ainvnm
					}
					if itran == 1 {
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
				*srnamt = "ZGTCON"
				golapack.Zgtcon(norm, &n, af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, &anorm, &rcond, work, &info)

				//              Check error code from ZGTCON.
				if info != 0 {
					Alaerh(path, []byte("ZGTCON"), &info, func() *int { y := 0; return &y }(), []byte{norm}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				}

				result.Set(6, Dget06(&rcond, &rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(6) >= (*thresh) {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						Alahd(path)
					}
					fmt.Printf(" NORM ='%c', N =%5d,           _type %2d, test(%2d) = %12.5f\n", norm, n, imat, 7, result.Get(6))
					nfail = nfail + 1
				}
				nrun = nrun + 1
			}

			//           Skip the remaining tests if the matrix is singular.
			if trfcon {
				goto label100
			}

			for irhs = 1; irhs <= (*nns); irhs++ {
				nrhs = (*nsval)[irhs-1]

				//              Generate NRHS random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &n, xact.Off(ix-1))
					ix = ix + lda
				}

				for itran = 1; itran <= 3; itran++ {
					trans = transs[itran-1]
					if itran == 1 {
						rcondc = rcondo
					} else {
						rcondc = rcondi
					}

					//                 Set the right hand side.
					golapack.Zlagtm(trans, &n, &nrhs, &one, a, a.Off(m+1-1), a.Off(n+m+1-1), xact.CMatrix(lda, opts), &lda, &zero, b.CMatrix(lda, opts), &lda)

					//+    TEST 2
					//              Solve op(A) * X = B and compute the residual.
					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)
					*srnamt = "ZGTTRS"
					golapack.Zgttrs(trans, &n, &nrhs, af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, x.CMatrix(lda, opts), &lda, &info)

					//              Check error code from ZGTTRS.
					if info != 0 {
						Alaerh(path, []byte("ZGTTRS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
					Zgtt02(trans, &n, &nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, result.GetPtr(1))

					//+    TEST 3
					//              Check solution from generated exact solution.
					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

					//+    TESTS 4, 5, and 6
					//              Use iterative refinement to improve the solution.
					*srnamt = "ZGTRFS"
					golapack.Zgtrfs(trans, &n, &nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), af, af.Off(m+1-1), af.Off(n+m+1-1), af.Off(n+2*m+1-1), iwork, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

					//              Check error code from ZGTRFS.
					if info != 0 {
						Alaerh(path, []byte("ZGTRFS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
					Zgtt05(trans, &n, &nrhs, a, a.Off(m+1-1), a.Off(n+m+1-1), b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(4))

					//              Print information about the tests that did not pass the
					//              threshold.
					for k = 2; k <= 6; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" TRANS='%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) = %12.5f\n", trans, n, nrhs, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 5
				}
			}
		label100:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
