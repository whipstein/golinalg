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

// Dchkgb tests DGBTRF, -TRS, -RFS, and -CON
func Dchkgb(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nns *int, nsval *[]int, thresh *float64, tsterr *bool, a *mat.Vector, la *int, afac *mat.Vector, lafac *int, b, x, xact, work, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var trfcon, zerot bool
	var dist, norm, trans, _type, xtype byte
	var ainvnm, anorm, anormi, anormo, cndnum, one, rcond, rcondc, rcondi, rcondo, zero float64
	var i, i1, i2, ikl, iku, im, imat, in, inb, info, ioff, irhs, itran, izero, j, k, kl, koff, ku, lda, ldafac, ldb, m, mode, n, nb, nerrs, nfail, nimat, nkl, nku, nrhs, nrun, ntran, ntypes int

	transs := make([]byte, 3)
	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	klval := make([]int, 4)
	kuval := make([]int, 4)

	one = 1.0
	zero = 0.0
	ntypes = 8
	ntran = 3

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3], transs[0], transs[1], transs[2] = 1988, 1989, 1990, 1991, 'N', 'T', 'C'

	//     Initialize constants and the random number seed.
	path := []byte("DGB")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrge(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

	//     Initialize the first value for the lower and upper bandwidths.
	klval[0] = 0
	kuval[0] = 0

	//     Do for each value of M in MVAL
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]

		//        Set values to use for the lower bandwidth.
		klval[1] = m + (m+1)/4

		//        KLVAL( 2 ) = MAX( M-1, 0 )
		klval[2] = (3*m - 1) / 4
		klval[3] = (m + 1) / 4

		//        Do for each value of N in NVAL
		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
			xtype = 'N'

			//           Set values to use for the upper bandwidth.
			kuval[1] = n + (n+1)/4

			//           KUVAL( 2 ) = MAX( N-1, 0 )
			kuval[2] = (3*n - 1) / 4
			kuval[3] = (n + 1) / 4

			//           Set limits on the number of loop iterations.
			nkl = minint(m+1, 4)
			if n == 0 {
				nkl = 2
			}
			nku = minint(n+1, 4)
			if m == 0 {
				nku = 2
			}
			nimat = ntypes
			if m <= 0 || n <= 0 {
				nimat = 1
			}

			for ikl = 1; ikl <= nkl; ikl++ {
				//              Do for KL = 0, (5*M+1)/4, (3M-1)/4, and (M+1)/4. This
				//              order makes it easier to skip redundant values for small
				//              values of M.
				kl = klval[ikl-1]
				for iku = 1; iku <= nku; iku++ {
					//                 Do for KU = 0, (5*N+1)/4, (3N-1)/4, and (N+1)/4. This
					//                 order makes it easier to skip redundant values for
					//                 small values of N.
					ku = kuval[iku-1]

					//                 Check that A and AFAC are big enough to generate this
					//                 matrix.
					lda = kl + ku + 1
					ldafac = 2*kl + ku + 1
					if (lda*n) > (*la) || (ldafac*n) > (*lafac) {
						t.Fail()
						if nfail == 0 && nerrs == 0 {
							Alahd(path)
						}
						if n*(kl+ku+1) > (*la) {
							fmt.Printf(" *** In DCHKGB, LA=%5d is too small for M=%5d, N=%5d, KL=%4d, KU=%4d\n ==> Increase LA to at least %5d\n", *la, m, n, kl, ku, n*(kl+ku+1))
							nerrs = nerrs + 1
						}
						if n*(2*kl+ku+1) > (*lafac) {
							fmt.Printf(" *** In DCHKGB, LAFAC=%5d is too small for M=%5d, N=%5d, KL=%4d, KU=%4d\n ==> Increase LAFAC to at least %5d\n", *lafac, m, n, kl, ku, n*(2*kl+ku+1))
							nerrs = nerrs + 1
						}
						goto label130
					}

					for imat = 1; imat <= nimat; imat++ {
						//                    Do the tests only if DOTYPE( IMAT ) is true.
						if !(*dotype)[imat-1] {
							goto label120
						}

						//                    Skip types 2, 3, or 4 if the matrix size is too
						//                    small.
						zerot = imat >= 2 && imat <= 4
						if zerot && n < imat-1 {
							goto label120
						}

						if !zerot || !(*dotype)[0] {
							//                       Set up parameters with DLATB4 and generate a
							//                       test matrix with DLATMS.
							Dlatb4(path, &imat, &m, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

							koff = maxint(1, ku+2-n)
							for i = 1; i <= koff-1; i++ {
								a.Set(i-1, zero)
							}
							*srnamt = "DLATMS"
							matgen.Dlatms(&m, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'Z', a.MatrixOff(koff-1, lda, opts), &lda, work, &info)

							//                       Check the error code from DLATMS.
							if info != 0 {
								Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
								goto label120
							}
						} else if izero > 0 {
							//                       Use the same matrix for types 3 and 4 as for
							//                       _type 2 by copying back the zeroed out column.
							goblas.Dcopy(toPtr(i2-i1+1), b, func() *int { y := 1; return &y }(), a.Off(ioff+i1-1), func() *int { y := 1; return &y }())
						}

						//                    For types 2, 3, and 4, zero one or more columns of
						//                    the matrix to test that INFO is returned correctly.
						izero = 0
						if zerot {
							if imat == 2 {
								izero = 1
							} else if imat == 3 {
								izero = minint(m, n)
							} else {
								izero = minint(m, n)/2 + 1
							}
							ioff = (izero - 1) * lda
							if imat < 4 {
								//                          Store the column to be zeroed out in B.
								i1 = maxint(1, ku+2-izero)
								i2 = minint(kl+ku+1, ku+1+(m-izero))
								goblas.Dcopy(toPtr(i2-i1+1), a.Off(ioff+i1-1), func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }())

								for i = i1; i <= i2; i++ {
									a.Set(ioff+i-1, zero)
								}
							} else {
								for j = izero; j <= n; j++ {
									for i = maxint(1, ku+2-j); i <= minint(kl+ku+1, ku+1+(m-j)); i++ {
										a.Set(ioff+i-1, zero)
									}
									ioff = ioff + lda
								}
							}
						}

						//                    These lines, if used in place of the calls in the
						//                    loop over INB, cause the code to bomb on a Sun
						//                    SPARCstation.
						//
						//                     ANORMO = DLANGB( 'O', N, KL, KU, A, LDA, RWORK )
						//                     ANORMI = DLANGB( 'I', N, KL, KU, A, LDA, RWORK )
						//
						//                    Do for each blocksize in NBVAL
						for inb = 1; inb <= (*nnb); inb++ {
							nb = (*nbval)[inb-1]
							Xlaenv(1, nb)

							//                       Compute the LU factorization of the band matrix.
							if m > 0 && n > 0 {
								golapack.Dlacpy('F', toPtr(kl+ku+1), &n, a.Matrix(lda, opts), &lda, afac.MatrixOff(kl+1-1, ldafac, opts), &ldafac)
							}
							*srnamt = "DGBTRF"
							golapack.Dgbtrf(&m, &n, &kl, &ku, afac.Matrix(ldafac, opts), &ldafac, iwork, &info)

							//                       Check error code from DGBTRF.
							if info != izero {
								Alaerh(path, []byte("DGBTRF"), &info, &izero, []byte{' '}, &m, &n, &kl, &ku, &nb, &imat, &nfail, &nerrs)
							}
							trfcon = false

							//+    TEST 1
							//                       Reconstruct matrix from factors and compute
							//                       residual.
							Dgbt01(&m, &n, &kl, &ku, a.Matrix(lda, opts), &lda, afac.Matrix(ldafac, opts), &ldafac, iwork, work, result.GetPtr(0))

							//                       Print information about the tests so far that
							//                       did not pass the threshold.
							if result.Get(0) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" M =%5d, N =%5d, KL=%5d, KU=%5d, NB =%4d, _type %1d, test(%1d)=%12.5f\n", m, n, kl, ku, nb, imat, 1, result.Get(0))
								nfail = nfail + 1
							}
							nrun = nrun + 1

							//                       Skip the remaining tests if this is not the
							//                       first block size or if M .ne. N.
							if inb > 1 || m != n {
								goto label110
							}

							anormo = golapack.Dlangb('O', &n, &kl, &ku, a.Matrix(lda, opts), &lda, rwork)
							anormi = golapack.Dlangb('I', &n, &kl, &ku, a.Matrix(lda, opts), &lda, rwork)

							if info == 0 {
								//                          Form the inverse of A so we can get a good
								//                          estimate of CNDNUM = norm(A) * norm(inv(A)).
								ldb = maxint(1, n)
								golapack.Dlaset('F', &n, &n, &zero, &one, work.Matrix(ldb, opts), &ldb)
								*srnamt = "DGBTRS"
								golapack.Dgbtrs('N', &n, &kl, &ku, &n, afac.Matrix(ldafac, opts), &ldafac, iwork, work.Matrix(ldb, opts), &ldb, &info)

								//                          Compute the 1-norm condition number of A.
								ainvnm = golapack.Dlange('O', &n, &n, work.Matrix(ldb, opts), &ldb, rwork)
								if anormo <= zero || ainvnm <= zero {
									rcondo = one
								} else {
									rcondo = (one / anormo) / ainvnm
								}

								//                          Compute the infinity-norm condition number of
								//                          A.
								ainvnm = golapack.Dlange('I', &n, &n, work.Matrix(ldb, opts), &ldb, rwork)
								if anormi <= zero || ainvnm <= zero {
									rcondi = one
								} else {
									rcondi = (one / anormi) / ainvnm
								}
							} else {
								//                          Do only the condition estimate if INFO.NE.0.
								trfcon = true
								rcondo = zero
								rcondi = zero
							}

							//                       Skip the solve tests if the matrix is singular.
							if trfcon {
								goto label90
							}

							for irhs = 1; irhs <= (*nns); irhs++ {
								nrhs = (*nsval)[irhs-1]
								xtype = 'N'

								for itran = 1; itran <= ntran; itran++ {
									trans = transs[itran-1]
									if itran == 1 {
										rcondc = rcondo
										norm = 'O'
									} else {
										rcondc = rcondi
										norm = 'I'
									}

									//+    TEST 2:
									//                             Solve and compute residual for A * X = B.
									*srnamt = "DLARHS"
									Dlarhs(path, &xtype, ' ', trans, &n, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb, &iseed, &info)
									xtype = 'C'
									golapack.Dlacpy('F', &n, &nrhs, b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb)

									*srnamt = "DGBTRS"
									golapack.Dgbtrs(trans, &n, &kl, &ku, &nrhs, afac.Matrix(ldafac, opts), &ldafac, iwork, x.Matrix(ldb, opts), &ldb, &info)

									//                             Check error code from DGBTRS.
									if info != 0 {
										Alaerh(path, []byte("DGBTRS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
									}

									golapack.Dlacpy('F', &n, &nrhs, b.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb)
									Dgbt02(trans, &m, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, x.Matrix(ldb, opts), &ldb, work.Matrix(ldb, opts), &ldb, result.GetPtr(1))

									//+    TEST 3:
									//                             Check solution from generated exact
									//                             solution.
									Dget04(&n, &nrhs, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, &rcondc, result.GetPtr(2))

									//+    TESTS 4, 5, 6:
									//                             Use iterative refinement to improve the
									//                             solution.
									*srnamt = "DGBRFS"
									golapack.Dgbrfs(trans, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, afac.Matrix(ldafac, opts), &ldafac, iwork, b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb, rwork, rwork.Off(nrhs+1-1), work, toSlice(iwork, n+1-1), &info)

									//                             Check error code from DGBRFS.
									if info != 0 {
										Alaerh(path, []byte("DGBRFS"), &info, func() *int { y := 0; return &y }(), []byte{trans}, &n, &n, &kl, &ku, &nrhs, &imat, &nfail, &nerrs)
									}

									Dget04(&n, &nrhs, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, &rcondc, result.GetPtr(3))
									Dgbt05(trans, &n, &kl, &ku, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, x.Matrix(ldb, opts), &ldb, xact.Matrix(ldb, opts), &ldb, rwork, rwork.Off(nrhs+1-1), result.Off(4))
									for k = 2; k <= 6; k++ {
										if result.Get(k-1) >= (*thresh) {
											t.Fail()
											if nfail == 0 && nerrs == 0 {
												Alahd(path)
											}
											fmt.Printf(" TRANS='%c', N=%5d, KL=%5d, KU=%5d, NRHS=%3d, _type %1d, test(%1d)=%12.5f\n", trans, n, kl, ku, nrhs, imat, k, result.Get(k-1))
											nfail = nfail + 1
										}
									}
									nrun = nrun + 5
								}
							}

							//+    TEST 7:
							//                          Get an estimate of RCOND = 1/CNDNUM.
						label90:
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
								*srnamt = "DGBCON"
								golapack.Dgbcon(norm, &n, &kl, &ku, afac.Matrix(ldafac, opts), &ldafac, iwork, &anorm, &rcond, work, toSlice(iwork, n+1-1), &info)

								//                             Check error code from DGBCON.
								if info != 0 {
									Alaerh(path, []byte("DGBCON"), &info, func() *int { y := 0; return &y }(), []byte{norm}, &n, &n, &kl, &ku, toPtr(-1), &imat, &nfail, &nerrs)
								}

								result.Set(6, Dget06(&rcond, &rcondc))

								//                          Print information about the tests that did
								//                          not pass the threshold.
								if result.Get(6) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" NORM ='%c', N=%5d, KL=%5d, KU=%5d,           _type %1d, test(%1d)=%12.5f\n", norm, n, kl, ku, imat, 7, result.Get(6))
									nfail = nfail + 1
								}
								nrun = nrun + 1
							}

						label110:
						}
					label120:
					}
				label130:
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 28938
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
