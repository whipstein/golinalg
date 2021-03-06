package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkge tests Zgetrf, -TRI, -TRS, -RFS, and -CON.
func zchkge(dotype []bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, nns int, nsval []int, thresh float64, tsterr bool, nmax int, a, afac, ainv, b, x, xact, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, _type, xtype byte
	var trans mat.MatTrans
	var ainvnm, anorm, anormi, anormo, cndnum, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, im, imat, in, inb, info, ioff, irhs, itran, izero, k, kl, ku, lda, lwork, m, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntypes int
	var err error

	path := "Zge"
	alasumStart(path)
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
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	xlaenv(1, 1)
	if tsterr {
		zerrge(path, t)
	}
	(*infot) = 0
	xlaenv(2, 2)

	//     Do for each value of M in MVAL
	for im = 1; im <= nm; im++ {
		m = mval[im-1]
		lda = max(1, m)

		//        Do for each value of N in NVAL
		for in = 1; in <= nn; in++ {
			n = nval[in-1]
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

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with Zlatms.
				_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, m, n)

				*srnamt = "Zlatms"
				if err = matgen.Zlatms(m, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a.CMatrix(lda, opts), work); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, m, n, -1, -1, -1, imat, nfail, nerrs)
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
							a.Set(ioff+i-1, complex(zero, 0))
						}
					} else {
						golapack.Zlaset(Full, m, n-izero+1, complex(zero, 0), complex(zero, 0), a.Off(ioff).CMatrix(lda, opts))
					}
				} else {
					izero = 0
				}

				//              These lines, if used in place of the calls in the DO 60
				//              loop, cause the code to bomb on a Sun SPARCstation.
				//
				//               ANORMO = ZLANGE( 'O', M, N, A, LDA, RWORK )
				//               ANORMI = ZLANGE( 'I', M, N, A, LDA, RWORK )
				//
				//              Do for each blocksize in NBVAL
				for inb = 1; inb <= nnb; inb++ {
					nb = nbval[inb-1]
					xlaenv(1, nb)

					//                 Compute the LU factorization of the matrix.
					golapack.Zlacpy(Full, m, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))
					*srnamt = "Zgetrf"
					if info, err = golapack.Zgetrf(m, n, afac.CMatrix(lda, opts), &iwork); err != nil || info != izero {
						t.Fail()
						nerrs = alaerh(path, "Zgetrf", info, 0, []byte{' '}, m, n, -1, -1, nb, imat, nfail, nerrs)
					}
					trfcon = false

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Zlacpy(Full, m, n, afac.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
					*result.GetPtr(0) = zget01(m, n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts), &iwork, rwork)
					nt = 1

					//+    TEST 2
					//                 Form the inverse if the factorization was successful
					//                 and compute the residual.
					if m == n && info == 0 {
						golapack.Zlacpy(Full, n, n, afac.CMatrix(lda, opts), ainv.CMatrix(lda, opts))
						*srnamt = "Zgetri"
						nrhs = nsval[0]
						lwork = nmax * max(3, nrhs)
						if info, err = golapack.Zgetri(n, ainv.CMatrix(lda, opts), &iwork, work, lwork); err != nil || info != 0 {
							t.Fail()
							nerrs = alaerh(path, "Zgetri", info, 0, []byte{' '}, n, n, -1, -1, nb, imat, nfail, nerrs)
						}

						//                    Compute the residual for the matrix times its
						//                    inverse.  Also compute the 1-norm condition number
						//                    of A.
						rcondo, *result.GetPtr(1) = zget03(n, a.CMatrix(lda, opts), ainv.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)
						anormo = golapack.Zlange('O', m, n, a.CMatrix(lda, opts), rwork)

						//                    Compute the infinity-norm condition number of A.
						anormi = golapack.Zlange('I', m, n, a.CMatrix(lda, opts), rwork)
						ainvnm = golapack.Zlange('I', n, n, ainv.CMatrix(lda, opts), rwork)
						if anormi <= zero || ainvnm <= zero {
							rcondi = one
						} else {
							rcondi = (one / anormi) / ainvnm
						}
						nt = 2
					} else {
						//                    Do only the condition estimate if INFO > 0.
						trfcon = true
						anormo = golapack.Zlange('O', m, n, a.CMatrix(lda, opts), rwork)
						anormi = golapack.Zlange('I', m, n, a.CMatrix(lda, opts), rwork)
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
							fmt.Printf(" M = %5d, n=%5d, NB =%4d, _type %2d, test(%2d) =%12.5f\n", m, n, nb, imat, k, result.Get(k-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Skip the remaining tests if this is not the first
					//                 block size or if M .ne. N.  Skip the solve tests if
					//                 the matrix is singular.
					if inb > 1 || m != n {
						goto label90
					}
					if trfcon {
						goto label70
					}

					for irhs = 1; irhs <= nns; irhs++ {
						nrhs = nsval[irhs-1]
						xtype = 'N'

						for _, trans = range mat.IterMatTrans() {
							if trans == NoTrans {
								rcondc = rcondo
							} else {
								rcondc = rcondi
							}

							//+    TEST 3
							//                       Solve and compute residual for A * X = B.
							*srnamt = "zlarhs"
							if err = zlarhs(path, xtype, Full, trans, n, n, kl, ku, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
								panic(err)
							}
							xtype = 'C'

							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))
							*srnamt = "Zgetrs"
							if err = golapack.Zgetrs(trans, n, nrhs, afac.CMatrix(lda, opts), &iwork, x.CMatrix(lda, opts)); err != nil {
								panic(err)
							}

							//                       Check error code from Zgetrs.
							if info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Zgetrs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							golapack.Zlacpy(Full, n, nrhs, b.CMatrix(lda, opts), work.CMatrix(lda, opts))
							*result.GetPtr(2) = zget02(trans, n, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), work.CMatrix(lda, opts), rwork)

							//+    TEST 4
							//                       Check solution from generated exact solution.
							*result.GetPtr(3) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)

							//+    TESTS 5, 6, and 7
							//                       Use iterative refinement to improve the
							//                       solution.
							*srnamt = "Zgerfs"
							if err = golapack.Zgerfs(trans, n, nrhs, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), &iwork, b.CMatrix(lda, opts), x.CMatrix(lda, opts), rwork, rwork.Off(nrhs), work, rwork.Off(2*nrhs)); err != nil {
								t.Fail()
								nerrs = alaerh(path, "Zgerfs", info, 0, []byte{trans.Byte()}, n, n, -1, -1, nrhs, imat, nfail, nerrs)
							}

							*result.GetPtr(4) = zget04(n, nrhs, x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rcondc)
							zget07(trans, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(lda, opts), x.CMatrix(lda, opts), xact.CMatrix(lda, opts), rwork, true, rwork.Off(nrhs), result.Off(5))

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
						*srnamt = "Zgecon"
						if rcond, err = golapack.Zgecon(norm, n, afac.CMatrix(lda, opts), anorm, work, rwork); err != nil {
							t.Fail()
							nerrs = alaerh(path, "Zgecon", info, 0, []byte{norm}, n, n, -1, -1, -1, imat, nfail, nerrs)
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
