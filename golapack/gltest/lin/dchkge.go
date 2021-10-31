package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkge tests DGETRF, -TRI, -TRS, -RFS, and -CON.
func dchkge(dotype []bool, mval, nval, nbval, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, _type, xtype byte
	var trans mat.MatTrans
	var ainvnm, anorm, anormi, anormo, cndnum, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, imat, inb, info, ioff, itran, izero, k, kl, ku, lda, lwork, m, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error
	var _iwork []int
	var _result *float64

	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 11

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dge"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	xlaenv(1, 1)
	if tsterr {
		derrge(path, t)
	}
	(*infot) = 0
	xlaenv(2, 2)

	//     Do for each value of M in MVAL
	for _, m = range mval {
		lda = max(1, m)

		//        Do for each value of N in NVAL
		for _, n = range nval {
			xtype = 'N'
			nimat = ntypes
			if m <= 0 || n <= 0 {
				nimat = 1
			}

			for imat = 1; imat <= nimat; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !dotype[imat-1] {
					goto label100
				}

				//              Skip types 5, 6, or 7 if the matrix size is too small.
				zerot = imat >= 5 && imat <= 7
				if zerot && n < imat-4 {
					goto label100
				}

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, m, n)

				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(m, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte{' '}, m, n, -1, -1, -1, imat, nfail, nerrs)
					goto label100
				}

				//              For types 5-7, zero one or more columns of the matrix to
				//              test that INFO is returned correctly.
				if zerot {
					if imat == 5 {
						izero = 1
					} else if imat == 6 {
						izero = min(m, n)
					} else {
						izero = min(m, n)/2 + 1
					}
					ioff = (izero - 1) * lda
					if imat < 7 {
						for i = 1; i <= m; i++ {
							a.Set(ioff+i-1, zero)
						}
					} else {
						golapack.Dlaset(Full, m, n-izero+1, zero, zero, a.MatrixOff(ioff, lda, opts))
					}
				} else {
					izero = 0
				}

				//              These lines, if used in place of the calls in the DO 60
				//              loop, cause the code to bomb on a Sun SPARCstation.
				//
				//               ANORMO = DLANGE( 'O', M, N, A, LDA, RWORK )
				//               ANORMI = DLANGE( 'I', M, N, A, LDA, RWORK )
				//
				//              Do for each blocksize in NBVAL
				for inb, nb = range nbval {
					xlaenv(1, nb)

					//                 Compute the LU factorization of the matrix.
					golapack.Dlacpy(Full, m, n, a.Matrix(lda, opts), afac.Matrix(lda, opts))
					*srnamt = "Dgetrf"
					if info, err = golapack.Dgetrf(m, n, afac.Matrix(lda, opts), &iwork); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Dgetrf", info, izero, []byte{' '}, m, n, -1, -1, nb, imat, nfail, nerrs)
					}
					trfcon = false

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Dlacpy(Full, m, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
					result.Set(0, dget01(m, n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), iwork, rwork))
					nt = 1

					//+    TEST 2
					//                 Form the inverse if the factorization was successful
					//                 and compute the residual.
					if m == n && info == 0 {
						golapack.Dlacpy(Full, n, n, afac.Matrix(lda, opts), ainv.Matrix(lda, opts))
						*srnamt = "Dgetri"
						nrhs = nsval[0]
						lwork = nmax * max(3, nrhs)
						if info, err = golapack.Dgetri(n, ainv.Matrix(lda, opts), iwork, work.Matrix(lwork, opts)); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Dgetri", info, 0, []byte{' '}, n, n, -1, -1, nb, imat, nfail, nerrs)
						}

						//                    Compute the residual for the matrix times its
						//                    inverse.  Also compute the 1-norm condition number
						//                    of A.
						_result = result.GetPtr(1)
						if rcondo, *_result, err = dget03(n, a.Matrix(lda, opts), ainv.Matrix(lda, opts), work.Matrix(lda, opts), rwork); err != nil {
							panic(err)
						}
						anormo = golapack.Dlange('O', m, n, a.Matrix(lda, opts), rwork)

						//                    Compute the infinity-norm condition number of A.
						anormi = golapack.Dlange('I', m, n, a.Matrix(lda, opts), rwork)
						ainvnm = golapack.Dlange('I', n, n, ainv.Matrix(lda, opts), rwork)
						if anormi <= zero || ainvnm <= zero {
							rcondi = one
						} else {
							rcondi = (one / anormi) / ainvnm
						}
						nt = 2
					} else {
						//                    Do only the condition estimate if INFO > 0.
						trfcon = true
						anormo = golapack.Dlange('O', m, n, a.Matrix(lda, opts), rwork)
						anormi = golapack.Dlange('I', m, n, a.Matrix(lda, opts), rwork)
						rcondo = zero
						rcondi = zero
					}

					//                 Print information about the tests so far that did not
					//                 pass the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" m=%5d, n=%5d, nb=%4d, _type %2d, test(%2d) =%12.5f\n", m, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Skip the remaining tests if this is not the first
					//                 block size or if M .ne. N.  Skip the solve tests if
					//                 the matrix is singular.
					if inb > 0 || m != n {
						goto label90
					}
					if trfcon {
						goto label70
					}

					for _, nrhs = range nsval {
						xtype = 'N'

						for itran, trans = range mat.IterMatTrans() {
							if itran == 0 {
								rcondc = rcondo
							} else {
								rcondc = rcondi
							}

							//+    TEST 3
							//                       Solve and compute residual for A * X = B.
							*srnamt = "Dlarhs"
							if err = Dlarhs(path, xtype, Full, trans, n, n, kl, ku, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'

							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))
							*srnamt = "Dgetrs"
							if err = golapack.Dgetrs(trans, n, nrhs, afac.Matrix(lda, opts), iwork, x.Matrix(lda, opts)); err != nil {
								t.Fail()
								nerrs = alaerh(path, "Dgetrs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							golapack.Dlacpy(Full, n, nrhs, b.Matrix(lda, opts), work.Matrix(lda, opts))
							result.Set(2, dget02(trans, n, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), work.Matrix(lda, opts), rwork))

							//+    TEST 4
							//                       Check solution from generated exact solution.
							result.Set(3, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))

							//+    TESTS 5, 6, and 7
							//                       Use iterative refinement to improve the
							//                       solution.
							*srnamt = "Dgerfs"
							if err = golapack.Dgerfs(trans, n, nrhs, a.Matrix(lda, opts), afac.Matrix(lda, opts), iwork, b.Matrix(lda, opts), x.Matrix(lda, opts), rwork, rwork.Off(nrhs), work, toSlice(&iwork, n)); err != nil {
								t.Fail()
								nerrs = alaerh(path, "Dgerfs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}
							for i, val := range _iwork {
								iwork[n+i] = val
							}

							result.Set(4, dget04(n, nrhs, x.Matrix(lda, opts), xact.Matrix(lda, opts), rcondc))
							dget07(trans, n, nrhs, a.Matrix(lda, opts), b.Matrix(lda, opts), x.Matrix(lda, opts), xact.Matrix(lda, opts), rwork, true, rwork.Off(nrhs), result.Off(5))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 7; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" trans=%s, n=%5d, nrhs=%3d, _type %2d, test(%2d) =%12.5f\n", trans, n, nrhs, imat, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 5
						}
					}

					//+    TEST 8
					//                    Get an estimate of RCOND = 1/CNDNUM.
				label70:
					;
					for itran = 1; itran <= 2; itran++ {
						if itran == 1 {
							anorm = anormo
							rcondc = rcondo
							norm = 'O'
						} else {
							anorm = anormi
							rcondc = rcondi
							norm = 'I'
						}
						*srnamt = "Dgecon"
						if rcond, err = golapack.Dgecon(norm, n, afac.Matrix(lda, opts), anorm, work, toSlice(&iwork, n)); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Dgecon", info, 0, []byte{norm}, n, n, -1, -1, -1, imat, nfail, nerrs)
						}
						for i, val := range _iwork {
							iwork[n+i] = val
						}

						result.Set(7, dget06(rcond, rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(7) >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" norm='%c', n=%5d,           _type %2d, test(%2d) =%12.5f\n", norm, n, imat, 8, result.Get(7))
							nfail++
						}
						nrun++
					}
				label90:
				}
			label100:
			}
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
