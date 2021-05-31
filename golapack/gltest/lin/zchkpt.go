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

// Zchkpt tests ZPTTRF, -TRS, -RFS, and -CON
func Zchkpt(dotype *[]bool, nn *int, nval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, a *mat.CVector, d *mat.Vector, e, b, x, xact, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var zerot bool
	var dist, _type, uplo byte
	var ainvnm, anorm, cond, dmax, one, rcond, rcondc, zero float64
	var i, ia, imat, in, info, irhs, iuplo, ix, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int

	uplos := make([]byte, 2)
	z := cvf(3)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 12
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3], uplos[0], uplos[1] = 0, 0, 0, 1, 'U', 'L'

	path := []byte("ZPT")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrgt(path, t)
	}
	(*infot) = 0

	for in = 1; in <= (*nn); in++ {
		//        Do for each value of N in NVAL.
		n = (*nval)[in-1]
		lda = maxint(1, n)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if n > 0 && !(*dotype)[imat-1] {
				goto label110
			}

			//           Set up parameters with ZLATB4.
			Zlatb4(path, &imat, &n, &n, &_type, &kl, &ku, &anorm, &mode, &cond, &dist)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Type 1-6:  generate a Hermitian tridiagonal matrix of
				//              known condition number in lower triangular band storage.
				*srnamt = "ZLATMS"
				matgen.Zlatms(&n, &n, dist, &iseed, _type, rwork, &mode, &cond, &anorm, &kl, &ku, 'B', a.CMatrix(2, opts), func() *int { y := 2; return &y }(), work, &info)

				//              Check the error code from ZLATMS.
				if info != 0 {
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
					goto label110
				}
				izero = 0

				//              Copy the matrix to D and E.
				ia = 1
				for i = 1; i <= n-1; i++ {
					d.Set(i-1, a.GetRe(ia-1))
					e.Set(i-1, a.Get(ia+1-1))
					ia = ia + 2
				}
				if n > 0 {
					d.Set(n-1, a.GetRe(ia-1))
				}
			} else {
				//              Type 7-12:  generate a diagonally dominant matrix with
				//              unknown condition number in the vectors D and E.
				if !zerot || !(*dotype)[6] {
					//                 Let E be complex, D real, with values from [-1,1].
					golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &n, d)
					golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(n-1), e)

					//                 Make the tridiagonal matrix diagonally dominant.
					if n == 1 {
						d.Set(0, math.Abs(d.Get(0)))
					} else {
						d.Set(0, math.Abs(d.Get(0))+e.GetMag(0))
						d.Set(n-1, math.Abs(d.Get(n-1))+e.GetMag(n-1-1))
						for i = 2; i <= n-1; i++ {
							d.Set(i-1, math.Abs(d.Get(i-1))+e.GetMag(i-1)+e.GetMag(i-1-1))
						}
					}

					//                 Scale D and E so the maximum element is ANORM.
					ix = goblas.Idamax(&n, d, func() *int { y := 1; return &y }())
					dmax = d.Get(ix - 1)
					goblas.Dscal(&n, toPtrf64(anorm/dmax), d, func() *int { y := 1; return &y }())
					goblas.Zdscal(toPtr(n-1), toPtrf64(anorm/dmax), e, func() *int { y := 1; return &y }())

				} else if izero > 0 {
					//                 Reuse the last matrix by copying back the zeroed out
					//                 elements.
					if izero == 1 {
						d.Set(0, z.GetRe(1))
						if n > 1 {
							e.Set(0, z.Get(2))
						}
					} else if izero == n {
						e.Set(n-1-1, z.Get(0))
						d.Set(n-1, z.GetRe(1))
					} else {
						e.Set(izero-1-1, z.Get(0))
						d.Set(izero-1, z.GetRe(1))
						e.Set(izero-1, z.Get(2))
					}
				}

				//              For types 8-10, set one row and column of the matrix to
				//              zero.
				izero = 0
				if imat == 8 {
					izero = 1
					z.Set(1, d.GetCmplx(0))
					d.Set(0, zero)
					if n > 1 {
						z.Set(2, e.Get(0))
						e.SetRe(0, zero)
					}
				} else if imat == 9 {
					izero = n
					if n > 1 {
						z.Set(0, e.Get(n-1-1))
						e.SetRe(n-1-1, zero)
					}
					z.SetRe(1, d.Get(n-1))
					d.Set(n-1, zero)
				} else if imat == 10 {
					izero = (n + 1) / 2
					if izero > 1 {
						z.Set(0, e.Get(izero-1-1))
						z.Set(2, e.Get(izero-1))
						e.SetRe(izero-1-1, zero)
						e.SetRe(izero-1, zero)
					}
					z.SetRe(1, d.Get(izero-1))
					d.Set(izero-1, zero)
				}
			}

			goblas.Dcopy(&n, d, func() *int { y := 1; return &y }(), d.Off(n+1-1), func() *int { y := 1; return &y }())
			if n > 1 {
				goblas.Zcopy(toPtr(n-1), e, func() *int { y := 1; return &y }(), e.Off(n+1-1), func() *int { y := 1; return &y }())
			}

			//+    TEST 1
			//           Factor A as L*D*L' and compute the ratio
			//              norm(L*D*L' - A) / (n * norm(A) * EPS )
			golapack.Zpttrf(&n, d.Off(n+1-1), e.Off(n+1-1), &info)

			//           Check error code from ZPTTRF.
			if info != izero {
				t.Fail()
				Alaerh(path, []byte("ZPTTRF"), &info, &izero, []byte{' '}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
				goto label110
			}

			if info > 0 {
				rcondc = zero
				goto label100
			}

			Zptt01(&n, d, e, d.Off(n+1-1), e.Off(n+1-1), work, result.GetPtr(0))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(0) >= (*thresh) {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					Alahd(path)
				}
				fmt.Printf(" N =%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 1, result.Get(0))
				nfail = nfail + 1
			}
			nrun = nrun + 1

			//           Compute RCONDC = 1 / (norm(A) * norm(inv(A))
			//
			//           Compute norm(A).
			anorm = golapack.Zlanht('1', &n, d, e)

			//           Use ZPTTRS to solve for one column at a time of inv(A),
			//           computing the maximum column sum as we go.
			ainvnm = zero
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					x.SetRe(j-1, zero)
				}
				x.SetRe(i-1, one)
				golapack.Zpttrs('L', &n, func() *int { y := 1; return &y }(), d.Off(n+1-1), e.Off(n+1-1), x.CMatrix(lda, opts), &lda, &info)
				ainvnm = maxf64(ainvnm, goblas.Dzasum(&n, x, func() *int { y := 1; return &y }()))
			}
			rcondc = one / maxf64(one, anorm*ainvnm)

			for irhs = 1; irhs <= (*nns); irhs++ {
				nrhs = (*nsval)[irhs-1]

				//           Generate NRHS random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &n, xact.Off(ix-1))
					ix = ix + lda
				}

				for iuplo = 1; iuplo <= 2; iuplo++ {
					//              Do first for UPLO = 'U', then for UPLO = 'L'.
					uplo = uplos[iuplo-1]

					//              Set the right hand side.
					Zlaptm(uplo, &n, &nrhs, &one, d, e, xact.CMatrix(lda, opts), &lda, &zero, b.CMatrix(lda, opts), &lda)

					//+    TEST 2
					//              Solve A*x = b and compute the residual.
					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)
					golapack.Zpttrs(uplo, &n, &nrhs, d.Off(n+1-1), e.Off(n+1-1), x.CMatrix(lda, opts), &lda, &info)

					//              Check error code from ZPTTRS.
					if info != 0 {
						Alaerh(path, []byte("ZPTTRS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					golapack.Zlacpy('F', &n, &nrhs, b.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda)
					Zptt02(uplo, &n, &nrhs, d, e, x.CMatrix(lda, opts), &lda, work.CMatrix(lda, opts), &lda, result.GetPtr(1))

					//+    TEST 3
					//              Check solution from generated exact solution.
					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(2))

					//+    TESTS 4, 5, and 6
					//              Use iterative refinement to improve the solution.
					*srnamt = "ZPTRFS"
					golapack.Zptrfs(uplo, &n, &nrhs, d, e, d.Off(n+1-1), e.Off(n+1-1), b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), work, rwork.Off(2*nrhs+1-1), &info)

					//              Check error code from ZPTRFS.
					if info != 0 {
						Alaerh(path, []byte("ZPTRFS"), &info, func() *int { y := 0; return &y }(), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
					}

					Zget04(&n, &nrhs, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, &rcondc, result.GetPtr(3))
					Zptt05(&n, &nrhs, d, e, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, rwork, rwork.Off(nrhs+1-1), result.Off(4))

					//              Print information about the tests that did not pass the
					//              threshold.
					for k = 2; k <= 6; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" UPLO = '%c', N =%5d, NRHS =%3d, _type %2d, test %2d, ratio = %12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 5

				}
			}

			//+    TEST 7
			//           Estimate the reciprocal of the condition number of the
			//           matrix.
		label100:
			;
			*srnamt = "ZPTCON"
			golapack.Zptcon(&n, d.Off(n+1-1), e.Off(n+1-1), &anorm, &rcond, rwork, &info)

			//           Check error code from ZPTCON.
			if info != 0 {
				t.Fail()
				Alaerh(path, []byte("ZPTCON"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
			}

			result.Set(6, Dget06(&rcond, &rcondc))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(6) >= (*thresh) {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					Alahd(path)
				}
				fmt.Printf(" N =%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 7, result.Get(6))
				nfail = nfail + 1
			}
			nrun = nrun + 1
		label110:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
