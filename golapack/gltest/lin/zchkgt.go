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

// zchkgt tests Zgttrf, -TRS, -RFS, and -CON
func zchkgt(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, a, af, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, _type byte
	var trans mat.MatTrans
	var ainvnm, anorm, cond, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, imat, in, info, irhs, itran, ix, izero, j, k, kl, koff, ku, lda, m, mode, n, nerrs, nfail, nimat, nrhs, nrun, ntypes int
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

	path := "Zgt"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrge(path, t)
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
				goto label100
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
					goto label100
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
					//                 Generate a matrix with elements whose real and
					//                 imaginary parts are from [-1,1].
					golapack.Zlarnv(2, &iseed, n+2*m, a)
					if anorm != one {
						goblas.Zdscal(n+2*m, anorm, a.Off(0, 1))
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
			goblas.Zcopy(n+2*m, a.Off(0, 1), af.Off(0, 1))
			*srnamt = "Zgttrf"
			if info, err = golapack.Zgttrf(n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork); err != nil || info != izero {
				nerrs = alaerh(path, "Zgttrf", info, 0, []byte{' '}, n, n, 1, 1, -1, imat, nfail, nerrs)
			}
			trfcon = info != 0

			*result.GetPtr(0) = zgtt01(n, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, work.CMatrix(lda, opts), rwork)

			//           Print the test ratio if it is .GE. THRESH.
			if result.Get(0) >= thresh {
				t.Fail()
				if nfail == 0 && nerrs == 0 {
					alahd(path)
				}
				fmt.Printf("            n=%5d,           _type %2d, test(%2d) = %12.5f\n", n, imat, 1, result.Get(0))
				nfail++
			}
			nrun++

			for _, trans = range mat.IterMatTrans() {
				if trans == NoTrans {
					norm = 'O'
				} else {
					norm = 'I'
				}
				anorm = golapack.Zlangt(norm, n, a, a.Off(m), a.Off(n+m))

				if !trfcon {
					//                 Use Zgttrs to solve for one column at a time of
					//                 inv(A), computing the maximum column sum as we go.
					ainvnm = zero
					for i = 1; i <= n; i++ {
						for j = 1; j <= n; j++ {
							x.SetRe(j-1, zero)
						}
						x.SetRe(i-1, one)
						if err = golapack.Zgttrs(trans, n, 1, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, x.CMatrix(lda, opts)); err != nil {
							panic(err)
						}
						ainvnm = math.Max(ainvnm, goblas.Dzasum(n, x.Off(0, 1)))
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
				*srnamt = "Zgtcon"
				if rcond, err = golapack.Zgtcon(norm, n, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, anorm, work); err != nil {
					nerrs = alaerh(path, "Zgtcon", info, 0, []byte{norm}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				result.Set(6, dget06(rcond, rcondc))

				//              Print the test ratio if it is .GE. THRESH.
				if result.Get(6) >= thresh {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					fmt.Printf(" norm='%c', n=%5d,           _type %2d, test(%2d) = %12.5f\n", norm, n, imat, 7, result.Get(6))
					nfail++
				}
				nrun++
			}

			//           Skip the remaining tests if the matrix is singular.
			if trfcon {
				goto label100
			}

			for irhs = 1; irhs <= nns; irhs++ {
				nrhs = nsval[irhs-1]

				//              Generate nrhs random solution vectors.
				ix = 1
				for j = 1; j <= nrhs; j++ {
					golapack.Zlarnv(2, &iseed, n, xact.Off(ix-1))
					ix = ix + lda
				}

				for _, trans = range mat.IterMatTrans() {
					if trans == NoTrans {
						rcondc = rcondo
					} else {
						rcondc = rcondi
					}

					//                 Set the right hand side.
					golapack.Zlagtm(trans, n, nrhs, one, a, a.Off(m), a.Off(n+m), xact.CMatrix(lda, opts), zero, b.CMatrix(lda, opts))

					//+    TEST 2
					//              Solve op(A) * X = B and compute the residual.
					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))
					*srnamt = "Zgttrs"
					if err = golapack.Zgttrs(trans, n, nrhs, af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, x.CMatrix(lda, opts)); err != nil {
						nerrs = alaerh(path, "Zgttrs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
					*result.GetPtr(1) = zgtt02(trans, n, nrhs, a, a.Off(m), a.Off(n+m), x.CMatrix(lda, opts), work.CMatrix(lda, opts))

					//+    TEST 3
					//              Check solution from generated exact solution.
					*result.GetPtr(2) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

					//+    TESTS 4, 5, and 6
					//              Use iterative refinement to improve the solution.
					*srnamt = "Zgtrfs"
					if err = golapack.Zgtrfs(trans, n, nrhs, a, a.Off(m), a.Off(n+m), af, af.Off(m), af.Off(n+m), af.Off(n+2*m), &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
						nerrs = alaerh(path, "Zgtrfs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
					zgtt05(trans, n, nrhs, a, a.Off(m), a.Off(n+m), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(4))

					//              Print information about the tests that did not pass the
					//              threshold.
					for k = 2; k <= 6; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" trans=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) = %12.5f\n", trans, n, nrhs, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 5
				}
			}
		label100:
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
