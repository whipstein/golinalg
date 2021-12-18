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

// zchkpt tests Zpttrf, -TRS, -RFS, and -CON
func zchkpt(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, a *mat.CVector, d *mat.Vector, e, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var zerot bool
	var dist, _type byte
	var uplo mat.MatUplo
	var ainvnm, anorm, cond, dmax, one, rcond, rcondc, zero float64
	var i, ia, imat, in, info, irhs, ix, izero, j, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int
	var err error

	z := cvf(3)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 12
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 0, 0, 0, 1

	path := "Zpt"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrgt(path, t)
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

			//           Set up parameters with ZLATB4.
			_type, kl, ku, anorm, mode, cond, dist = zlatb4(path, imat, n, n)

			zerot = imat >= 8 && imat <= 10
			if imat <= 6 {
				//              Type 1-6:  generate a Hermitian tridiagonal matrix of
				//              known condition number in lower triangular band storage.
				*srnamt = "Zlatms"
				if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cond, anorm, kl, ku, 'B', a.CMatrix(2, opts), work); err != nil {
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, n, n, kl, ku, -1, imat, nfail, nerrs)
					goto label110
				}
				izero = 0

				//              Copy the matrix to D and E.
				ia = 1
				for i = 1; i <= n-1; i++ {
					d.Set(i-1, a.GetRe(ia-1))
					e.Set(i-1, a.Get(ia))
					ia = ia + 2
				}
				if n > 0 {
					d.Set(n-1, a.GetRe(ia-1))
				}
			} else {
				//              Type 7-12:  generate a diagonally dominant matrix with
				//              unknown condition number in the vectors D and E.
				if !zerot || !dotype[6] {
					//                 Let E be complex, D real, with values from [-1,1].
					golapack.Dlarnv(2, &iseed, n, d)
					golapack.Zlarnv(2, &iseed, n-1, e)

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
					ix = d.Iamax(n, 1)
					dmax = d.Get(ix - 1)
					d.Scal(n, anorm/dmax, 1)
					e.Dscal(n-1, anorm/dmax, 1)

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

			d.Off(n).Copy(n, d, 1, 1)
			if n > 1 {
				e.Off(n).Copy(n-1, e, 1, 1)
			}

			//+    TEST 1
			//           Factor A as L*D*L' and compute the ratio
			//              norm(L*D*L' - A) / (n * norm(A) * EPS )
			if info, err = golapack.Zpttrf(n, d.Off(n), e.Off(n)); err != nil || info != izero {
				t.Fail()
				nerrs = alaerh(path, "Zpttrf", info, 0, []byte{' '}, n, n, -1, -1, -1, imat, nfail, nerrs)
				goto label110
			}

			if info > 0 {
				rcondc = zero
				goto label100
			}

			*result.GetPtr(0) = zptt01(n, d, e, d.Off(n), e.Off(n), work)

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(0) >= thresh {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				fmt.Printf(" n=%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 1, result.Get(0))
				nfail++
			}
			nrun++

			//           Compute RCONDC = 1 / (norm(A) * norm(inv(A))
			//
			//           Compute norm(A).
			anorm = golapack.Zlanht('1', n, d, e)

			//           Use Zpttrs to solve for one column at a time of inv(A),
			//           computing the maximum column sum as we go.
			ainvnm = zero
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					x.SetRe(j-1, zero)
				}
				x.SetRe(i-1, one)
				if err = golapack.Zpttrs(Lower, n, 1, d.Off(n), e.Off(n), x.CMatrix(lda, opts)); err != nil {
					panic(err)
				}
				ainvnm = math.Max(ainvnm, x.Asum(n, 1))
			}
			rcondc = one / math.Max(one, anorm*ainvnm)

			for irhs = 1; irhs <= nns; irhs++ {
				nrhs = nsval[irhs-1]

				//           Generate NRHS random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Zlarnv(2, &iseed, n, xact.Off(ix-1))
					ix = ix + lda
				}

				for _, uplo = range mat.IterMatUplo(false) {
					//              Do first for uplo='U', then for uplo='L'.

					//              Set the right hand side.
					zlaptm(uplo, n, nrhs, one, d, e, xact.CMatrix(lda, opts), zero, b.CMatrix(lda, opts))

					//+    TEST 2
					//              Solve A*x = b and compute the residual.
					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))
					if err = golapack.Zpttrs(uplo, n, nrhs, d.Off(n), e.Off(n), x.CMatrix(lda, opts)); err != nil {
						nerrs = alaerh(path, "Zpttrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
					*result.GetPtr(1) = zptt02(uplo, n, nrhs, d, e, x.CMatrix(lda, opts), work.CMatrix(lda, opts))

					//+    TEST 3
					//              Check solution from generated exact solution.
					*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

					//+    TESTS 4, 5, and 6
					//              Use iterative refinement to improve the solution.
					*srnamt = "Zptrfs"
					if err = golapack.Zptrfs(uplo, n, nrhs, d, e, d.Off(n), e.Off(n), b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
						nerrs = alaerh(path, "Zptrfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
					zptt05(n, nrhs, d, e, b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

					//              Print information about the tests that did not pass the
					//              threshold.
					for k = 2; k <= 6; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test %2d, ratio = %12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 5

				}
			}

			//+    TEST 7
			//           Estimate the reciprocal of the condition number of the
			//           matrix.
		label100:
			;
			*srnamt = "Zptcon"
			if rcond, err = golapack.Zptcon(n, d.Off(n), e.Off(n), anorm, rwork); err != nil {
				t.Fail()
				nerrs = alaerh(path, "Zptcon", info, 0, []byte{' '}, n, n, -1, -1, -1, imat, nfail, nerrs)
			}

			result.Set(6, dget06(rcond, rcondc))

			//           Print the test ratio if greater than or equal to THRESH.
			if result.Get(6) >= thresh {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				fmt.Printf(" n=%5d, _type %2d, test %2d, ratio = %12.5f\n", n, imat, 7, result.Get(6))
				nfail++
			}
			nrun++
		label110:
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
