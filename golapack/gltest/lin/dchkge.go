package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dchkge tests DGETRF, -TRI, -TRS, -RFS, and -CON.
func Dchkge(dotype []bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr bool, nmax *int, a, afac, ainv, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, trans, _type, xtype byte
	var ainvnm, anorm, anormi, anormo, cndnum, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, im, imat, in, inb, info, ioff, irhs, itran, izero, k, kl, ku, lda, lwork, m, mode, n, nb, nerrs, nfail, nimat, nrhs, nrun, nt, ntran, ntypes int

	transs := make([]byte, 3)
	result := vf(8)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 11
	ntran = 3

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3], transs[0], transs[1], transs[2] = 1988, 1989, 1990, 1991, 'N', 'T', 'C'

	//     Initialize constants and the random number seed.
	path := []byte("DGE")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	Xlaenv(1, 1)
	if tsterr {
		Derrge(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

	//     Do for each value of M in MVAL
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]
		lda = max(1, m)

		//        Do for each value of N in NVAL
		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
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
				Dlatb4(path, &imat, &m, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&m, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'N', a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
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
						golapack.Dlaset('F', &m, toPtr(n-izero+1), &zero, &zero, a.MatrixOff(ioff, lda, opts), &lda)
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
				for inb = 1; inb <= (*nnb); inb++ {
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)

					//                 Compute the LU factorization of the matrix.
					golapack.Dlacpy('F', &m, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
					*srnamt = "DGETRF"
					golapack.Dgetrf(&m, &n, afac.Matrix(lda, opts), &lda, iwork, &info)

					//                 Check error code from DGETRF.
					if info != izero {
						Alaerh(path, []byte("DGETRF"), &info, &izero, []byte{' '}, &m, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
					}
					trfcon = false

					//+    TEST 1
					//                 Reconstruct matrix from factors and compute residual.
					golapack.Dlacpy('F', &m, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
					Dget01(&m, &n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, iwork, rwork, result.GetPtr(0))
					nt = 1

					//+    TEST 2
					//                 Form the inverse if the factorization was successful
					//                 and compute the residual.
					if m == n && info == 0 {
						golapack.Dlacpy('F', &n, &n, afac.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda)
						*srnamt = "DGETRI"
						nrhs = (*nsval)[0]
						lwork = (*nmax) * max(3, nrhs)
						golapack.Dgetri(&n, ainv.Matrix(lda, opts), &lda, iwork, work.Matrix(lwork, opts), &lwork, &info)

						//                    Check error code from DGETRI.
						if info != 0 {
							Alaerh(path, []byte("DGETRI"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
						}

						//                    Compute the residual for the matrix times its
						//                    inverse.  Also compute the 1-norm condition number
						//                    of A.
						Dget03(&n, a.Matrix(lda, opts), &lda, ainv.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, &rcondo, result.GetPtr(1))
						anormo = golapack.Dlange('O', &m, &n, a.Matrix(lda, opts), &lda, rwork)

						//                    Compute the infinity-norm condition number of A.
						anormi = golapack.Dlange('I', &m, &n, a.Matrix(lda, opts), &lda, rwork)
						ainvnm = golapack.Dlange('I', &n, &n, ainv.Matrix(lda, opts), &lda, rwork)
						if anormi <= zero || ainvnm <= zero {
							rcondi = one
						} else {
							rcondi = (one / anormi) / ainvnm
						}
						nt = 2
					} else {
						//                    Do only the condition estimate if INFO > 0.
						trfcon = true
						anormo = golapack.Dlange('O', &m, &n, a.Matrix(lda, opts), &lda, rwork)
						anormi = golapack.Dlange('I', &m, &n, a.Matrix(lda, opts), &lda, rwork)
						rcondo = zero
						rcondi = zero
					}

					//                 Print information about the tests so far that did not
					//                 pass the threshold.
					for k = 1; k <= nt; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" M = %5d, N =%5d, NB =%4d, _type %2d, test(%2d) =%12.5f\n", m, n, nb, imat, k, result.Get(k-1))
							nfail = nfail + 1
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

					for irhs = 1; irhs <= (*nns); irhs++ {
						nrhs = (*nsval)[irhs-1]
						xtype = 'N'

						for itran = 1; itran <= ntran; itran++ {
							trans = transs[itran-1]
							if itran == 1 {
								rcondc = rcondo
							} else {
								rcondc = rcondi
							}

							//+    TEST 3
							//                       Solve and compute residual for A * X = B.
							*srnamt = "DLARHS"
							Dlarhs(path, &xtype, ' ', trans, &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)
							xtype = 'C'

							golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)
							*srnamt = "DGETRS"
							golapack.Dgetrs(trans, &n, &nrhs, afac.Matrix(lda, opts), &lda, iwork, x.Matrix(lda, opts), &lda, &info)

							//                       Check error code from DGETRS.
							if info != 0 {
								Alaerh(path, []byte("DGETRS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}

							golapack.Dlacpy('F', &n, &nrhs, b.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda)
							Dget02(trans, &n, &n, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, work.Matrix(lda, opts), &lda, rwork, result.GetPtr(2))

							//+    TEST 4
							//                       Check solution from generated exact solution.
							Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(3))

							//+    TESTS 5, 6, and 7
							//                       Use iterative refinement to improve the
							//                       solution.
							*srnamt = "DGERFS"
							golapack.Dgerfs(trans, &n, &nrhs, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, iwork, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, rwork, rwork.Off(nrhs), work, toSlice(iwork, n), &info)

							//                       Check error code from DGERFS.
							if info != 0 {
								Alaerh(path, []byte("DGERFS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, toPtr(-1), toPtr(-1), &nrhs, &imat, &nfail, &nerrs)
							}

							Dget04(&n, &nrhs, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, &rcondc, result.GetPtr(4))
							Dget07(trans, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, rwork, true, rwork.Off(nrhs), result.Off(5))

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 7; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" TRANS='%c', N =%5d, NRHS=%3d, _type %2d, test(%2d) =%12.5f\n", trans, n, nrhs, imat, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 5
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
						*srnamt = "DGECON"
						golapack.Dgecon(norm, &n, afac.Matrix(lda, opts), &lda, &anorm, &rcond, work, toSlice(iwork, n), &info)

						//                       Check error code from DGECON.
						if info != 0 {
							Alaerh(path, []byte("DGECON"), &info, func() *int { y := 0; return &y }(), []byte{norm}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						}

						result.Set(7, Dget06(&rcond, &rcondc))

						//                    Print information about the tests that did not pass
						//                    the threshold.
						if result.Get(7) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" NORM ='%c', N =%5d,           _type %2d, test(%2d) =%12.5f\n", norm, n, imat, 8, result.Get(7))
							nfail = nfail + 1
						}
						nrun = nrun + 1
					}
				label90:
				}
			label100:
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
