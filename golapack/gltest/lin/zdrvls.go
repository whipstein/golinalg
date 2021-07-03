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

// Zdrvls tests the least squares driver routines ZGELS, ZGETSLS, ZGELSS, ZGELSY
// and ZGELSD.
func Zdrvls(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nns *int, nsval *[]int, nnb *int, nbval *[]int, nxval *[]int, thresh *float64, tsterr *bool, a, copya, b, copyb, c *mat.CVector, s, copys *mat.Vector, nout *int, t *testing.T) {
	var trans byte
	var cone, czero complex128
	var eps, norma, normb, one, rcond, zero float64
	var crank, i, im, imb, in, inb, info, ins, irank, iscale, itran, itype, j, k, lda, ldb, ldwork, liwork, lrwork, lrworkZgelsd, lrworkZgelss, lrworkZgelsy, lwlsy, lwork, lworkZgels, lworkZgelsd, lworkZgelss, lworkZgelsy, lworkZgetsls, m, mb, mmax, mnmin, n, nb, ncols, nerrs, nfail, nmax, nrhs, nrows, nrun, nsmax, rank, smlsiz int
	var err error
	_ = err

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
	path := []byte("ZLS")
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
	Xlaenv(9, smlsiz)
	if *tsterr {
		Zerrls(path, t)
	}

	//     Print the header if NM = 0 or NN = 0 and THRESH = 0.
	if ((*nm) == 0 || (*nn) == 0) && (*thresh) == zero {
		t.Fail()
		Alahd(path)
	}
	(*infot) = 0

	//     Compute maximal workspace needed for all routines
	nmax = 0
	mmax = 0
	nsmax = 0
	for i = 1; i <= (*nm); i++ {
		if (*mval)[i-1] > mmax {
			mmax = (*mval)[i-1]
		}
	}
	for i = 1; i <= (*nn); i++ {
		if (*nval)[i-1] > nmax {
			nmax = (*nval)[i-1]
		}
	}
	for i = 1; i <= (*nns); i++ {
		if (*nsval)[i-1] > nsmax {
			nsmax = (*nsval)[i-1]
		}
	}
	m = mmax
	n = nmax
	nrhs = nsmax
	mnmin = maxint(minint(m, n), 1)

	//     Compute workspace needed for routines
	//     ZQRT14, ZQRT17 (two side cases), ZQRT15 and ZQRT12
	lwork = maxint(1, (m+n)*nrhs, (n+nrhs)*(m+2), (m+nrhs)*(n+2), maxint(m+mnmin, nrhs*mnmin, 2*n+m), maxint(m*n+4*mnmin+maxint(m, n), m*n+2*mnmin+4*n))
	lrwork = 1
	liwork = 1

	rwork := vf(lrwork)

	//     Iterate through all test cases and compute necessary workspace
	//     sizes for ?GELS, ?GETSLS, ?GELSY, ?GELSS and ?GELSD routines.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]
		lda = maxint(1, m)
		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
			mnmin = maxint(minint(m, n), 1)
			ldb = maxint(1, m, n)
			for ins = 1; ins <= (*nns); ins++ {
				nrhs = (*nsval)[ins-1]
				for irank = 1; irank <= 2; irank++ {
					for iscale = 1; iscale <= 3; iscale++ {
						itype = (irank-1)*3 + iscale
						if (*dotype)[itype-1] {
							if irank == 1 {
								for itran = 1; itran <= 2; itran++ {
									if itran == 1 {
										trans = 'N'
									} else {
										trans = 'C'
									}

									//                             Compute workspace needed for ZGELS
									golapack.Zgels(trans, &m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, wq, toPtr(-1), &info)
									lworkZgels = int(wq.GetRe(0))
									//                             Compute workspace needed for ZGETSLS
									golapack.Zgetsls(trans, &m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, wq, toPtr(-1), &info)
									lworkZgetsls = int(wq.GetRe(0))
								}
							}
							//                       Compute workspace needed for ZGELSY
							golapack.Zgelsy(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, &iwq, &rcond, &crank, wq, toPtr(-1), rwork, &info)
							lworkZgelsy = int(wq.GetRe(0))
							lrworkZgelsy = 2 * n
							//                       Compute workspace needed for ZGELSS
							golapack.Zgelss(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, s, &rcond, &crank, wq, toPtr(-1), rwork, &info)
							lworkZgelss = int(wq.GetRe(0))
							lrworkZgelss = 5 * mnmin
							//                       Compute workspace needed for ZGELSD
							golapack.Zgelsd(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, s, &rcond, &crank, wq, toPtr(-1), rwq, &iwq, &info)
							lworkZgelsd = int(wq.GetRe(0))
							lrworkZgelsd = int(rwq.Get(0))
							//                       Compute LIWORK workspace needed for ZGELSY and ZGELSD
							liwork = maxint(liwork, n, iwq[0])
							//                       Compute LRWORK workspace needed for ZGELSY, ZGELSS and ZGELSD
							lrwork = maxint(lrwork, lrworkZgelsy, lrworkZgelss, lrworkZgelsd)
							//                       Compute LWORK workspace needed for all functions
							lwork = maxint(lwork, lworkZgels, lworkZgetsls, lworkZgelsy, lworkZgelss, lworkZgelsd)
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

	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]
		lda = maxint(1, m)

		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
			mnmin = maxint(minint(m, n), 1)
			ldb = maxint(1, m, n)
			mb = (mnmin + 1)

			for ins = 1; ins <= (*nns); ins++ {
				nrhs = (*nsval)[ins-1]

				for irank = 1; irank <= 2; irank++ {
					for iscale = 1; iscale <= 3; iscale++ {
						itype = (irank-1)*3 + iscale
						if !(*dotype)[itype-1] {
							goto label100
						}

						if irank == 1 {
							//                       Test ZGELS
							//
							//                       Generate a matrix of scaling _type ISCALE
							Zqrt13(&iscale, &m, &n, copya.CMatrix(lda, opts), &lda, &norma, &iseed)
							for inb = 1; inb <= (*nnb); inb++ {
								nb = (*nbval)[inb-1]
								Xlaenv(1, nb)
								Xlaenv(3, (*nxval)[inb-1])

								for itran = 1; itran <= 2; itran++ {
									if itran == 1 {
										trans = 'N'
										nrows = m
										ncols = n
									} else {
										trans = 'C'
										nrows = n
										ncols = m
									}
									ldwork = maxint(1, ncols)

									//                             Set up a consistent rhs
									if ncols > 0 {
										golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(ncols*nrhs), work)
										goblas.Zdscal(ncols*nrhs, one/float64(ncols), work, 1)
									}
									err = goblas.Zgemm(mat.TransByte(trans), NoTrans, nrows, nrhs, ncols, cone, copya.CMatrix(lda, opts), lda, work.CMatrix(ldwork, opts), ldwork, czero, b.CMatrix(ldb, opts), ldb)
									golapack.Zlacpy('F', &nrows, &nrhs, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb)

									//                             Solve LS or overdetermined system
									if m > 0 && n > 0 {
										golapack.Zlacpy('F', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
										golapack.Zlacpy('F', &nrows, &nrhs, copyb.CMatrix(ldb, opts), &ldb, b.CMatrix(ldb, opts), &ldb)
									}
									*srnamt = "ZGELS "
									golapack.Zgels(trans, &m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork, &info)

									if info != 0 {
										t.Fail()
										Alaerh(path, []byte("ZGELS "), &info, func() *int { y := 0; return &y }(), []byte{trans}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
									}

									//                             Check correctness of results
									ldwork = maxint(1, nrows)
									if nrows > 0 && nrhs > 0 {
										golapack.Zlacpy('F', &nrows, &nrhs, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), &ldb)
									}
									Zqrt16(trans, &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), &ldb, rwork, result.GetPtr(0))

									if (itran == 1 && m >= n) || (itran == 2 && m < n) {
										//                                Solving LS system
										result.Set(1, Zqrt17(trans, func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), work, &lwork))
									} else {
										//                                Solving overdetermined system
										result.Set(1, Zqrt14(trans, &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork))
									}

									//                             Print information about the tests that
									//                             did not pass the threshold.
									for k = 1; k <= 2; k++ {
										if result.Get(k-1) >= (*thresh) {
											t.Fail()
											if nfail == 0 && nerrs == 0 {
												Alahd(path)
											}
											fmt.Printf(" TRANS='%c', M=%5d, N=%5d, NRHS=%4d, NB=%4d, _type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, nb, itype, k, result.Get(k-1))
											nfail = nfail + 1
										}
									}
									nrun = nrun + 2
								}
							}

							//                       Test ZGETSLS
							//
							//                       Generate a matrix of scaling _type ISCALE
							Zqrt13(&iscale, &m, &n, copya.CMatrix(lda, opts), &lda, &norma, &iseed)
							for inb = 1; inb <= (*nnb); inb++ {
								mb = (*nbval)[inb-1]
								Xlaenv(1, mb)
								for imb = 1; imb <= (*nnb); imb++ {
									nb = (*nbval)[imb-1]
									Xlaenv(2, nb)

									for itran = 1; itran <= 2; itran++ {
										if itran == 1 {
											trans = 'N'
											nrows = m
											ncols = n
										} else {
											trans = 'C'
											nrows = n
											ncols = m
										}
										ldwork = maxint(1, ncols)

										//                             Set up a consistent rhs
										if ncols > 0 {
											golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(ncols*nrhs), work)
											goblas.Zscal(ncols*nrhs, complex(one/float64(ncols), 0), work, 1)
										}
										err = goblas.Zgemm(mat.TransByte(trans), NoTrans, nrows, nrhs, ncols, cone, copya.CMatrix(lda, opts), lda, work.CMatrix(ldwork, opts), ldwork, czero, b.CMatrix(ldb, opts), ldb)
										golapack.Zlacpy('F', &nrows, &nrhs, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb)

										//                             Solve LS or overdetermined system
										if m > 0 && n > 0 {
											golapack.Zlacpy('F', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
											golapack.Zlacpy('F', &nrows, &nrhs, copyb.CMatrix(ldb, opts), &ldb, b.CMatrix(ldb, opts), &ldb)
										}
										*srnamt = "ZGETSLS "
										golapack.Zgetsls(trans, &m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork, &info)
										if info != 0 {
											t.Fail()
											Alaerh(path, []byte("ZGETSLS "), &info, func() *int { y := 0; return &y }(), []byte{trans}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
										}

										//                             Check correctness of results
										ldwork = maxint(1, nrows)
										if nrows > 0 && nrhs > 0 {
											golapack.Zlacpy('F', &nrows, &nrhs, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), &ldb)
										}
										Zqrt16(trans, &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), &ldb, work2, result.GetPtr(14))

										if (itran == 1 && m >= n) || (itran == 2 && m < n) {
											//                                Solving LS system
											result.Set(15, Zqrt17(trans, func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), work, &lwork))
										} else {
											//                                Solving overdetermined system
											result.Set(15, Zqrt14(trans, &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork))
										}

										//                             Print information about the tests that
										//                             did not pass the threshold.
										for k = 15; k <= 16; k++ {
											if result.Get(k-1) >= (*thresh) {
												t.Fail()
												if nfail == 0 && nerrs == 0 {
													Alahd(path)
												}
												fmt.Printf(" TRANS='%c M=%5d, N=%5d, NRHS=%4d, MB=%4d, NB=%4d, _type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, mb, nb, itype, k, result.Get(k-1))
												nfail = nfail + 1
											}
										}
										nrun = nrun + 2
									}
								}
							}
						}

						//                    Generate a matrix of scaling _type ISCALE and rank
						//                    _type IRANK.
						Zqrt15(&iscale, &irank, &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, copyb.CMatrix(ldb, opts), &ldb, copys, &rank, &norma, &normb, &iseed, work, &lwork)

						//                    workspace used: maxint(M+minint(M,N),NRHS*minint(M,N),2*N+M)
						ldwork = maxint(1, m)

						//                    Loop for testing different block sizes.
						for inb = 1; inb <= (*nnb); inb++ {
							nb = (*nbval)[inb-1]
							Xlaenv(1, nb)
							Xlaenv(3, (*nxval)[inb-1])

							//                       Test ZGELSY
							//
							//                       ZGELSY:  Compute the minimum-norm solution
							//                       X to minint( norm( A * X - B ) )
							//                       using the rank-revealing orthogonal
							//                       factorization.
							golapack.Zlacpy('F', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, b.CMatrix(ldb, opts), &ldb)

							//                       Initialize vector iwork.
							for j = 1; j <= n; j++ {
								iwork[j-1] = 0
							}

							*srnamt = "ZGELSY"
							golapack.Zgelsy(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, &iwork, &rcond, &crank, work, &lwlsy, rwork, &info)
							if info != 0 {
								Alaerh(path, []byte("ZGELSY"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       workspace used: 2*MNMIN+NB*NB+NB*maxint(N,NRHS)
							//
							//                       Test 3:  Compute relative error in svd
							//                                workspace: M*N + 4*minint(M,N) + maxint(M,N)
							result.Set(2, Zqrt12(&crank, &crank, a.CMatrix(lda, opts), &lda, copys, work, &lwork, rwork))

							//                       Test 4:  Compute error in solution
							//                                workspace:  M*NRHS + M
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork)
							Zqrt16('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork, rwork, result.GetPtr(3))

							//                       Test 5:  Check norm of r'*A
							//                                workspace: NRHS*(M+N)
							result.Set(4, zero)
							if m > crank {
								result.Set(4, Zqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), work, &lwork))
							}

							//                       Test 6:  Check if x is in the rowspace of A
							//                                workspace: (M+NRHS)*(N+2)
							result.Set(5, zero)

							if n > crank {
								result.Set(5, Zqrt14('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Test ZGELSS
							//
							//                       ZGELSS:  Compute the minimum-norm solution
							//                       X to minint( norm( A * X - B ) )
							//                       using the SVD.
							golapack.Zlacpy('F', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, b.CMatrix(ldb, opts), &ldb)
							*srnamt = "ZGELSS"
							golapack.Zgelss(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, s, &rcond, &crank, work, &lwork, rwork, &info)

							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZGELSS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       workspace used: 3*minint(m,n) +
							//                                       maxint(2*minint(m,n),nrhs,maxint(m,n))
							//
							//                       Test 7:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(mnmin, -one, copys, 1, s, 1)
								result.Set(6, goblas.Dasum(mnmin, s, 1)/goblas.Dasum(mnmin, copys, 1)/(eps*float64(mnmin)))
							} else {
								result.Set(6, zero)
							}

							//                       Test 8:  Compute error in solution
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork)
							Zqrt16('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork, rwork, result.GetPtr(7))

							//                       Test 9:  Check norm of r'*A
							result.Set(8, zero)
							if m > crank {
								result.Set(8, Zqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), work, &lwork))
							}

							//                       Test 10:  Check if x is in the rowspace of A
							result.Set(9, zero)
							if n > crank {
								result.Set(9, Zqrt14('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Test ZGELSD
							//
							//                       ZGELSD:  Compute the minimum-norm solution X
							//                       to minint( norm( A * X - B ) ) using a
							//                       divide and conquer SVD.
							Xlaenv(9, 25)

							golapack.Zlacpy('F', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, b.CMatrix(ldb, opts), &ldb)

							*srnamt = "ZGELSD"
							golapack.Zgelsd(&m, &n, &nrhs, a.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, s, &rcond, &crank, work, &lwork, rwork, &iwork, &info)
							if info != 0 {
								t.Fail()
								Alaerh(path, []byte("ZGELSD"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       Test 11:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(mnmin, -one, copys, 1, s, 1)
								result.Set(10, goblas.Dasum(mnmin, s, 1)/goblas.Dasum(mnmin, copys, 1)/(eps*float64(mnmin)))
							} else {
								result.Set(10, zero)
							}

							//                       Test 12:  Compute error in solution
							golapack.Zlacpy('F', &m, &nrhs, copyb.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork)
							Zqrt16('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work.CMatrix(ldwork, opts), &ldwork, rwork, result.GetPtr(11))

							//                       Test 13:  Check norm of r'*A
							result.Set(12, zero)
							if m > crank {
								result.Set(12, Zqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, copyb.CMatrix(ldb, opts), &ldb, c.CMatrix(ldb, opts), work, &lwork))
							}

							//                       Test 14:  Check if x is in the rowspace of A
							result.Set(13, zero)
							if n > crank {
								result.Set(13, Zqrt14('N', &m, &n, &nrhs, copya.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 14; k++ {
								if result.Get(k-1) >= (*thresh) {
									t.Fail()
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									fmt.Printf(" M=%5d, N=%5d, NRHS=%4d, NB=%4d, _type%2d, test(%2d)=%12.5f\n", m, n, nrhs, nb, itype, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 12

						}
					label100:
					}
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
