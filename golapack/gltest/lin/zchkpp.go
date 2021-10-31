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

// zchkpp tests Zpptrf, -TRI, -TRS, -RFS, and -CON
func zchkpp(dotype []bool, nn int, nval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var zerot bool
	var dist, packit, _type, xtype byte
	var uplo mat.MatUplo
	var anorm, cndnum, rcond, rcondc, zero float64
	var i, imat, in, info, ioff, irhs, iuplo, izero, k, kl, ku, lda, mode, n, nerrs, nfail, nimat, npp, nrhs, nrun, ntypes int
	var err error

	packs := make([]byte, 2)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	zero = 0.0
	ntypes = 9
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	packs[0], packs[1] = 'C', 'R'

	//     Initialize constants and the random number seed.
	path := "Zpp"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrpo(path, t)
	}
	(*infot) = 0

	//     Do for each value of N in NVAL
	for in = 1; in <= nn; in++ {
		n = nval[in-1]
		lda = max(n, 1)
		xtype = 'N'
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label100
			}

			//           Skip types 3, 4, or 5 if the matrix size is too small.
			zerot = imat >= 3 && imat <= 5
			if zerot && n < imat-2 {
				goto label100
			}

			//           Do first for uplo='U', then for uplo='L'
			for iuplo, uplo = range mat.IterMatUplo(false) {
				packit = packs[iuplo]

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with Zlatms.
				_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)

				*srnamt = "Zlatms"
				if err = matgen.Zlatms(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, packit, a.CMatrix(lda, opts), work); err != nil {
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					goto label90
				}

				//              For types 3-5, zero one row and column of the matrix to
				//              test that INFO is returned correctly.
				if zerot {
					if imat == 3 {
						izero = 1
					} else if imat == 4 {
						izero = n
					} else {
						izero = n/2 + 1
					}

					//                 Set row and column IZERO of A to 0.
					if uplo == Upper {
						ioff = (izero - 1) * izero / 2
						for i = 1; i <= izero-1; i++ {
							a.SetRe(ioff+i-1, zero)
						}
						ioff = ioff + izero
						for i = izero; i <= n; i++ {
							a.SetRe(ioff-1, zero)
							ioff = ioff + i
						}
					} else {
						ioff = izero
						for i = 1; i <= izero-1; i++ {
							a.SetRe(ioff-1, zero)
							ioff = ioff + n - i
						}
						ioff = ioff - izero
						for i = izero; i <= n; i++ {
							a.SetRe(ioff+i-1, zero)
						}
					}
				} else {
					izero = 0
				}

				//              Set the imaginary part of the diagonals.
				if uplo == Upper {
					zlaipd(n, a, 2, 1)
				} else {
					zlaipd(n, a, n, -1)
				}

				//              Compute the L*L' or U'*U factorization of the matrix.
				npp = n * (n + 1) / 2
				goblas.Zcopy(npp, a.Off(0, 1), afac.Off(0, 1))
				*srnamt = "Zpptrf"
				if info, err = golapack.Zpptrf(uplo, n, afac); err != nil || info != izero {
					t.Fail()
					nerrs = alaerh(path, "Zpptrf", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
					goto label90
				}

				//              Skip the tests if INFO is not 0.
				if info != 0 {
					goto label90
				}

				//+    TEST 1
				//              Reconstruct matrix from factors and compute residual.
				goblas.Zcopy(npp, afac.Off(0, 1), ainv.Off(0, 1))
				*result.GetPtr(0) = zppt01(uplo, n, a, ainv, rwork)

				//+    TEST 2
				//              Form the inverse and compute the residual.
				goblas.Zcopy(npp, afac.Off(0, 1), ainv.Off(0, 1))
				*srnamt = "Zpptri"
				if info, err = golapack.Zpptri(uplo, n, ainv); err != nil || info != 0 {
					t.Fail()
					nerrs = alaerh(path, "Zpptri", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				rcondc, *result.GetPtr(1) = zppt03(uplo, n, a, ainv, work.CMatrix(lda, opts), rwork)

				//              Print information about the tests that did not pass
				//              the threshold.
				for k = 1; k <= 2; k++ {
					if result.Get(k-1) >= thresh {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							alahd(path)
						}
						fmt.Printf(" uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, k, result.Get(k-1))
						nfail++
					}
				}
				nrun += 2

				for irhs = 1; irhs <= nns; irhs++ {
					nrhs = nsval[irhs-1]

					//+    TEST 3
					//              Solve and compute residual for  A * X = B.
					*srnamt = "zlarhs"
					if err = zlarhs(path, xtype, uplo, NoTrans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
						panic(err)
					}
					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))

					*srnamt = "Zpptrs"
					if err = golapack.Zpptrs(uplo, n, nrhs, afac, x.CMatrix(lda, opts)); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zpptrs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
					*result.GetPtr(2) = zppt02(uplo, n, nrhs, a, x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

					//+    TEST 4
					//              Check solution from generated exact solution.
					*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

					//+    TESTS 5, 6, and 7
					//              Use iterative refinement to improve the solution.
					*srnamt = "Zpprfs"
					if err = golapack.Zpprfs(uplo, n, nrhs, a, afac, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zpprfs", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
					}

					*result.GetPtr(4) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
					zppt05(uplo, n, nrhs, a, b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, rwork.Off(nrhs), result.Off(5))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 3; k <= 7; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", uplo, n, nrhs, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 5
				}

				//+    TEST 8
				//              Get an estimate of RCOND = 1/CNDNUM.
				anorm = golapack.Zlanhp('1', uplo, n, a, rwork)
				*srnamt = "Zppcon"
				if rcond, err = golapack.Zppcon(uplo, n, afac, anorm, work, rwork); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zppcon", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
				}

				result.Set(7, dget06(rcond, rcondc))

				//              Print the test ratio if greater than or equal to THRESH.
				if result.Get(7) >= thresh {
					t.Fail()
					if nfail == 0 && nerrs == 0 {
						alahd(path)
					}
					fmt.Printf(" uplo=%s, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", uplo, n, imat, 8, result.Get(7))
					nfail++
				}
				nrun++

			label90:
			}
		label100:
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
