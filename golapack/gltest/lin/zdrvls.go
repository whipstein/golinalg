package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zdrvls tests the least squares driver routines ZGELS, ZGETSLS, Zgelss, Zgelsy
// and Zgelsd.
func zdrvls(dotype []bool, nm int, mval []int, nn int, nval []int, nns int, nsval []int, nnb int, nbval []int, nxval []int, thresh float64, tsterr bool, a, copya, b, copyb, c *mat.CVector, s, copys *mat.Vector, t *testing.T) {
	var trans mat.MatTrans
	var cone, czero complex128
	var eps, one, rcond, zero float64
	var crank, i, im, imb, in, inb, info, ins, irank, iscale, itype, j, k, lda, ldb, ldwork, liwork, lrwork, lrworkZgelsd, lrworkZgelss, lrworkZgelsy, lwlsy, lwork, lworkZgels, lworkZgelsd, lworkZgelss, lworkZgelsy, lworkZgetsls, m, mb, mmax, mnmin, n, nb, ncols, nerrs, nfail, nmax, nrhs, nrows, nrun, nsmax, rank, smlsiz int
	var err error

	wq := cvf(1)
	result := vf(16)
	rwq := vf(1)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	iwq := make([]int, 1)

	smlsiz = 25
	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Zls"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	eps = golapack.Dlamch(Epsilon)

	//     Threshold for rank estimation
	rcond = math.Sqrt(eps) - (math.Sqrt(eps)-eps)/2

	//     Test the error exits
	xlaenv(9, smlsiz)
	if tsterr {
		zerrls(path, t)
	}

	//     Print the header if NM = 0 or NN = 0 and THRESH = 0.
	if (nm == 0 || nn == 0) && thresh == zero {
		t.Fail()
		alahd(path)
	}
	(*infot) = 0

	//     Compute maximal workspace needed for all routines
	nmax = 0
	mmax = 0
	nsmax = 0
	for i = 1; i <= nm; i++ {
		if mval[i-1] > mmax {
			mmax = mval[i-1]
		}
	}
	for i = 1; i <= nn; i++ {
		if nval[i-1] > nmax {
			nmax = nval[i-1]
		}
	}
	for i = 1; i <= nns; i++ {
		if nsval[i-1] > nsmax {
			nsmax = nsval[i-1]
		}
	}
	m = mmax
	n = nmax
	nrhs = nsmax
	mnmin = max(min(m, n), 1)

	//     Compute workspace needed for routines
	//     ZQRT14, ZQRT17 (two side cases), ZQRT15 and ZQRT12
	lwork = max(1, (m+n)*nrhs, (n+nrhs)*(m+2), (m+nrhs)*(n+2), max(m+mnmin, nrhs*mnmin, 2*n+m), max(m*n+4*mnmin+max(m, n), m*n+2*mnmin+4*n))
	lrwork = 1
	liwork = 1

	rwork := vf(lrwork)

	//     Iterate through all test cases and compute necessary workspace
	//     sizes for ?GELS, ?GETSLS, ?GELSY, ?GELSS and ?GELSD routines.
	for im = 1; im <= nm; im++ {
		m = mval[im-1]
		lda = max(1, m)
		for in = 1; in <= nn; in++ {
			n = nval[in-1]
			mnmin = max(min(m, n), 1)
			ldb = max(1, m, n)
			for ins = 1; ins <= nns; ins++ {
				nrhs = nsval[ins-1]
				for irank = 1; irank <= 2; irank++ {
					for iscale = 1; iscale <= 3; iscale++ {
						itype = (irank-1)*3 + iscale
						if dotype[itype-1] {
							if irank == 1 {
								for _, trans = range mat.IterMatTrans(false) {
									if trans == Trans {
										trans = ConjTrans
									}

									//                             Compute workspace needed for ZGELS
									if info, err = golapack.Zgels(trans, m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), wq, -1); err != nil {
										panic(err)
									}
									lworkZgels = int(wq.GetRe(0))
									//                             Compute workspace needed for ZGETSLS
									if info, err = golapack.Zgetsls(trans, m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), wq, -1); err != nil {
										panic(err)
									}
									lworkZgetsls = int(wq.GetRe(0))
								}
							}
							//                       Compute workspace needed for Zgelsy
							if crank, err = golapack.Zgelsy(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), &iwq, rcond, wq, -1, rwork); err != nil {
								panic(err)
							}
							lworkZgelsy = int(wq.GetRe(0))
							lrworkZgelsy = 2 * n
							//                       Compute workspace needed for Zgelss
							if crank, info, err = golapack.Zgelss(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), s, rcond, wq, -1, rwork); err != nil {
								panic(err)
							}
							lworkZgelss = int(wq.GetRe(0))
							lrworkZgelss = 5 * mnmin
							//                       Compute workspace needed for Zgelsd
							if crank, info, err = golapack.Zgelsd(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), s, rcond, wq, -1, rwq, &iwq); err != nil {
								panic(err)
							}
							lworkZgelsd = int(wq.GetRe(0))
							lrworkZgelsd = int(rwq.Get(0))
							//                       Compute LIWORK workspace needed for Zgelsy and Zgelsd
							liwork = max(liwork, n, iwq[0])
							//                       Compute LRWORK workspace needed for Zgelsy, Zgelss and Zgelsd
							lrwork = max(lrwork, lrworkZgelsy, lrworkZgelss, lrworkZgelsd)
							//                       Compute LWORK workspace needed for all functions
							lwork = max(lwork, lworkZgels, lworkZgetsls, lworkZgelsy, lworkZgelss, lworkZgelsd)
						}
					}
				}
			}
		}
	}

	lwlsy = lwork

	work := cvf(lwork)
	work2 := vf(2 * lwork)
	iwork := make([]int, liwork)
	rwork = vf(lrwork)

	for im = 1; im <= nm; im++ {
		m = mval[im-1]
		lda = max(1, m)

		for in = 1; in <= nn; in++ {
			n = nval[in-1]
			mnmin = max(min(m, n), 1)
			ldb = max(1, m, n)
			mb = (mnmin + 1)

			for ins = 1; ins <= nns; ins++ {
				nrhs = nsval[ins-1]

				for irank = 1; irank <= 2; irank++ {
					for iscale = 1; iscale <= 3; iscale++ {
						itype = (irank-1)*3 + iscale
						if !dotype[itype-1] {
							goto label100
						}

						if irank == 1 {
							//                       Test ZGELS
							//
							//                       Generate a matrix of scaling _type ISCALE
							_ = zqrt13(iscale, m, n, copya.CMatrix(lda, opts), &iseed)
							for inb = 1; inb <= nnb; inb++ {
								nb = nbval[inb-1]
								xlaenv(1, nb)
								xlaenv(3, nxval[inb-1])

								for _, trans = range mat.IterMatTrans(false) {
									if trans == NoTrans {
										nrows = m
										ncols = n
									} else {
										trans = ConjTrans
										nrows = n
										ncols = m
									}
									ldwork = max(1, ncols)

									//                             Set up a consistent rhs
									if ncols > 0 {
										golapack.Zlarnv(2, &iseed, ncols*nrhs, work)
										goblas.Zdscal(ncols*nrhs, one/float64(ncols), work.Off(0, 1))
									}
									if err = goblas.Zgemm(trans, NoTrans, nrows, nrhs, ncols, cone, copya.CMatrix(lda, opts), work.CMatrix(ldwork, opts), czero, b.CMatrix(ldb, opts)); err != nil {
										panic(err)
									}
									golapack.Zlacpy(Full, nrows, nrhs, b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts))

									//                             Solve LS or overdetermined system
									if m > 0 && n > 0 {
										golapack.Zlacpy(Full, m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts))
										golapack.Zlacpy(Full, nrows, nrhs, copyb.CMatrix(ldb, opts), b.CMatrix(ldb, opts))
									}
									*srnamt = "Zgels"
									if info, err = golapack.Zgels(trans, m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork); err != nil {
										panic(err)
									}

									if info != 0 {
										t.Fail()
										nerrs = alaerh(path, "Zgels", info, 0, []byte{trans.Byte()}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
									}

									//                             Check correctness of results
									ldwork = max(1, nrows)
									if nrows > 0 && nrhs > 0 {
										golapack.Zlacpy(Full, nrows, nrhs, copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts))
									}
									*result.GetPtr(0) = zqrt16(trans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), c.CMatrix(ldb, opts), rwork)

									if (trans == NoTrans && m >= n) || (trans == ConjTrans && m < n) {
										//                                Solving LS system
										result.Set(1, zqrt17(trans, 1, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work, lwork))
									} else {
										//                                Solving overdetermined system
										result.Set(1, zqrt14(trans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork))
									}

									//                             Print information about the tests that
									//                             did not pass the threshold.
									for k = 1; k <= 2; k++ {
										if result.Get(k-1) >= thresh {
											t.Fail()
											if nfail == 0 && nerrs == 0 {
												alahd(path)
											}
											fmt.Printf(" trans=%s, m=%5d, n=%5d, nrhs=%4d, nb=%4d, _type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, nb, itype, k, result.Get(k-1))
											nfail++
										}
									}
									nrun += 2
								}
							}

							//                       Test ZGETSLS
							//
							//                       Generate a matrix of scaling _type ISCALE
							_ = zqrt13(iscale, m, n, copya.CMatrix(lda, opts), &iseed)
							for inb = 1; inb <= nnb; inb++ {
								mb = nbval[inb-1]
								xlaenv(1, mb)
								for imb = 1; imb <= nnb; imb++ {
									nb = nbval[imb-1]
									xlaenv(2, nb)

									for _, trans = range mat.IterMatTrans(false) {
										if trans == NoTrans {
											nrows = m
											ncols = n
										} else {
											trans = ConjTrans
											nrows = n
											ncols = m
										}
										ldwork = max(1, ncols)

										//                             Set up a consistent rhs
										if ncols > 0 {
											golapack.Zlarnv(2, &iseed, ncols*nrhs, work)
											goblas.Zscal(ncols*nrhs, complex(one/float64(ncols), 0), work.Off(0, 1))
										}
										if err = goblas.Zgemm(trans, NoTrans, nrows, nrhs, ncols, cone, copya.CMatrix(lda, opts), work.CMatrix(ldwork, opts), czero, b.CMatrix(ldb, opts)); err != nil {
											panic(err)
										}
										golapack.Zlacpy(Full, nrows, nrhs, b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts))

										//                             Solve LS or overdetermined system
										if m > 0 && n > 0 {
											golapack.Zlacpy(Full, m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts))
											golapack.Zlacpy(Full, nrows, nrhs, copyb.CMatrix(ldb, opts), b.CMatrix(ldb, opts))
										}
										*srnamt = "Zgetsls"
										if info, err = golapack.Zgetsls(trans, m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork); err != nil || info != 0 {
											t.Fail()
											nerrs = alaerh(path, "Zgetsls", info, 0, []byte{trans.Byte()}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
										}

										//                             Check correctness of results
										ldwork = max(1, nrows)
										if nrows > 0 && nrhs > 0 {
											golapack.Zlacpy(Full, nrows, nrhs, copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts))
										}
										*result.GetPtr(14) = zqrt16(trans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work2)

										if (trans == NoTrans && m >= n) || (trans == ConjTrans && m < n) {
											//                                Solving LS system
											result.Set(15, zqrt17(trans, 1, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work, lwork))
										} else {
											//                                Solving overdetermined system
											result.Set(15, zqrt14(trans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork))
										}

										//                             Print information about the tests that
										//                             did not pass the threshold.
										for k = 15; k <= 16; k++ {
											if result.Get(k-1) >= thresh {
												t.Fail()
												if nfail == 0 && nerrs == 0 {
													alahd(path)
												}
												fmt.Printf(" trans=%s m=%5d, n=%5d, nrhs=%4d, MB=%4d, nb=%4d, _type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, mb, nb, itype, k, result.Get(k-1))
												nfail++
											}
										}
										nrun += 2
									}
								}
							}
						}

						//                    Generate a matrix of scaling _type ISCALE and rank
						//                    _type IRANK.
						if rank, _, _, err = zqrt15(iscale, irank, m, n, nrhs, copya.CMatrix(lda, opts), copyb.CMatrix(ldb, opts), copys, &iseed, work, lwork); err != nil {
							panic(err)
						}

						//                    workspace used: max(M+min(M,N),nrhs*min(M,N),2*N+M)
						ldwork = max(1, m)

						//                    Loop for testing different block sizes.
						for inb = 1; inb <= nnb; inb++ {
							nb = nbval[inb-1]
							xlaenv(1, nb)
							xlaenv(3, nxval[inb-1])

							//                       Test Zgelsy
							//
							//                       Zgelsy:  Compute the minimum-norm solution
							//                       X to min( norm( A * X - B ) )
							//                       using the rank-revealing orthogonal
							//                       factorization.
							golapack.Zlacpy(Full, m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts))
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), b.CMatrix(ldb, opts))

							//                       Initialize vector iwork.
							for j = 1; j <= n; j++ {
								iwork[j-1] = 0
							}

							*srnamt = "Zgelsy"
							if crank, err = golapack.Zgelsy(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), &iwork, rcond, work, lwlsy, rwork); err != nil {
								nerrs = alaerh(path, "Zgelsy", info, 0, []byte{' '}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
							}

							//                       workspace used: 2*MNMIN+nb*nb+nb*max(N,nrhs)
							//
							//                       Test 3:  Compute relative error in svd
							//                                workspace: M*N + 4*min(M,N) + max(M,N)
							result.Set(2, zqrt12(crank, crank, a.CMatrix(lda, opts), copys, work, lwork, rwork))

							//                       Test 4:  Compute error in solution
							//                                workspace:  M*nrhs + M
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), work.CMatrix(ldwork, opts))
							*result.GetPtr(3) = zqrt16(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work.CMatrix(ldwork, opts), rwork)

							//                       Test 5:  Check norm of r'*A
							//                                workspace: nrhs*(M+N)
							result.Set(4, zero)
							if m > crank {
								result.Set(4, zqrt17(NoTrans, 1, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work, lwork))
							}

							//                       Test 6:  Check if x is in the rowspace of A
							//                                workspace: (M+nrhs)*(N+2)
							result.Set(5, zero)

							if n > crank {
								result.Set(5, zqrt14(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork))
							}

							//                       Test Zgelss
							//
							//                       Zgelss:  Compute the minimum-norm solution
							//                       X to min( norm( A * X - B ) )
							//                       using the SVD.
							golapack.Zlacpy(Full, m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts))
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), b.CMatrix(ldb, opts))
							*srnamt = "Zgelss"
							if crank, info, err = golapack.Zgelss(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), s, rcond, work, lwork, rwork); err != nil {
								panic(err)
							}

							if info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Zgelss", info, 0, []byte{' '}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
							}

							//                       workspace used: 3*min(m,n) +
							//                                       max(2*min(m,n),nrhs,max(m,n))
							//
							//                       Test 7:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(mnmin, -one, copys.Off(0, 1), s.Off(0, 1))
								result.Set(6, goblas.Dasum(mnmin, s.Off(0, 1))/goblas.Dasum(mnmin, copys.Off(0, 1))/(eps*float64(mnmin)))
							} else {
								result.Set(6, zero)
							}

							//                       Test 8:  Compute error in solution
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), work.CMatrix(ldwork, opts))
							*result.GetPtr(7) = zqrt16(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work.CMatrix(ldwork, opts), rwork)

							//                       Test 9:  Check norm of r'*A
							result.Set(8, zero)
							if m > crank {
								result.Set(8, zqrt17(NoTrans, 1, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work, lwork))
							}

							//                       Test 10:  Check if x is in the rowspace of A
							result.Set(9, zero)
							if n > crank {
								result.Set(9, zqrt14(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork))
							}

							//                       Test Zgelsd
							//
							//                       Zgelsd:  Compute the minimum-norm solution X
							//                       to min( norm( A * X - B ) ) using a
							//                       divide and conquer SVD.
							xlaenv(9, 25)

							golapack.Zlacpy(Full, m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts))
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), b.CMatrix(ldb, opts))

							*srnamt = "Zgelsd"
							if crank, info, err = golapack.Zgelsd(m, n, nrhs, a.CMatrix(lda, opts), b.CMatrix(ldb, opts), s, rcond, work, lwork, rwork, &iwork); err != nil || info != 0 {
								t.Fail()
								nerrs = alaerh(path, "Zgelsd", info, 0, []byte{' '}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
							}

							//                       Test 11:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(mnmin, -one, copys.Off(0, 1), s.Off(0, 1))
								result.Set(10, goblas.Dasum(mnmin, s.Off(0, 1))/goblas.Dasum(mnmin, copys.Off(0, 1))/(eps*float64(mnmin)))
							} else {
								result.Set(10, zero)
							}

							//                       Test 12:  Compute error in solution
							golapack.Zlacpy(Full, m, nrhs, copyb.CMatrix(ldb, opts), work.CMatrix(ldwork, opts))
							*result.GetPtr(11) = zqrt16(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work.CMatrix(ldwork, opts), rwork)

							//                       Test 13:  Check norm of r'*A
							result.Set(12, zero)
							if m > crank {
								result.Set(12, zqrt17(NoTrans, 1, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), copyb.CMatrix(ldb, opts), c.CMatrix(ldb, opts), work, lwork))
							}

							//                       Test 14:  Check if x is in the rowspace of A
							result.Set(13, zero)
							if n > crank {
								result.Set(13, zqrt14(NoTrans, m, n, nrhs, copya.CMatrix(lda, opts), b.CMatrix(ldb, opts), work, lwork))
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 14; k++ {
								if result.Get(k-1) >= thresh {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									fmt.Printf(" m=%5d, n=%5d, nrhs=%4d, nb=%4d, _type%2d, test(%2d)=%12.5f\n", m, n, nrhs, nb, itype, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 12

						}
					label100:
					}
				}
			}
		}
	}

	//     Print a summary of the results.
	alasvm(path, nfail, nrun, nerrs)
}
