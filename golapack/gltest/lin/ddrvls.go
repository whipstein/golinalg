package lin

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
	"testing"
)

// Ddrvls tests the least squares driver routines DGELS, DGETSLS, DGELSS, DGELSY,
// and DGELSD.
func Ddrvls(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nns *int, nsval *[]int, nnb *int, nbval *[]int, nxval *[]int, thresh *float64, tsterr *bool, a, copya, b, copyb, c, s, copys *mat.Vector, nout *int, t *testing.T) {
	var trans byte
	var eps, norma, normb, one, rcond, zero float64
	var crank, i, im, imb, in, inb, info, ins, irank, iscale, itran, itype, j, k, lda, ldb, ldwork, liwork, lwlsy, lwork, lworkDgels, lworkDgelsd, lworkDgelss, lworkDgelsy, lworkDgetsls, m, mb, mmax, mnmin, n, nb, ncols, nerrs, nfail, nmax, nrhs, nrows, nrun, nsmax, rank, smlsiz int

	result := vf(16)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	iwq := make([]int, 1)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	// ntests = 16
	smlsiz = 25
	one = 1.0
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := []byte("DLS")
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
	Xlaenv(2, 2)
	Xlaenv(9, smlsiz)
	if *tsterr {
		Derrls(path, t)
	}

	//     Print the header if NM = 0 or NN = 0 and THRESH = 0.
	if ((*nm) == 0 || (*nn) == 0) && (*thresh) == zero {
		Alahd(path)
	}
	(*infot) = 0
	Xlaenv(2, 2)
	Xlaenv(9, smlsiz)

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
	wq := vf(1)

	//     Compute workspace needed for routines
	//     DQRT14, DQRT17 (two side cases), DQRT15 and DQRT12
	lwork = maxint(1, (m+n)*nrhs, (n+nrhs)*(m+2), (m+nrhs)*(n+2), maxint(m+mnmin, nrhs*mnmin, 2*n+m), maxint(m*n+4*mnmin+maxint(m, n), m*n+2*mnmin+4*n))
	liwork = 1

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
										trans = 'T'
									}

									//                             Compute workspace needed for DGELS
									golapack.Dgels(trans, &m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, wq, toPtr(-1), &info)
									lworkDgels = int(wq.Get(0))
									//                             Compute workspace needed for DGETSLS
									golapack.Dgetsls(trans, &m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, wq, toPtr(-1), &info)
									lworkDgetsls = int(wq.Get(0))
								}
							}
							//                       Compute workspace needed for DGELSY
							golapack.Dgelsy(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, &iwq, &rcond, &crank, wq, toPtr(-1), &info)
							lworkDgelsy = int(wq.Get(0))
							//                       Compute workspace needed for DGELSS
							golapack.Dgelss(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, s, &rcond, &crank, wq, toPtr(-1), &info)
							lworkDgelss = int(wq.Get(0))
							//                       Compute workspace needed for DGELSD
							golapack.Dgelsd(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, s, &rcond, &crank, wq, toPtr(-1), &iwq, &info)
							lworkDgelsd = int(wq.Get(0))
							//                       Compute LIWORK workspace needed for DGELSY and DGELSD
							liwork = maxint(liwork, n, iwq[0])
							//                       Compute LWORK workspace needed for all functions
							lwork = maxint(lwork, lworkDgels, lworkDgetsls, lworkDgelsy, lworkDgelss, lworkDgelsd)
						}
					}
				}
			}
		}
	}

	lwlsy = lwork

	work := vf(lwork)
	iwork := make([]int, liwork)

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
							goto label110
						}

						if irank == 1 {
							//                       Test DGELS
							//
							//                       Generate a matrix of scaling type ISCALE
							Dqrt13(&iscale, &m, &n, copya.Matrix(lda, opts), &lda, &norma, &iseed)
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
										trans = 'T'
										nrows = n
										ncols = m
									}
									ldwork = maxint(1, ncols)

									//                             Set up a consistent rhs
									if ncols > 0 {
										golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(ncols*nrhs), work)
										goblas.Dscal(toPtr(ncols*nrhs), toPtrf64(one/float64(ncols)), work, toPtr(1))
									}
									goblas.Dgemm(mat.TransByte(trans), NoTrans, &nrows, &nrhs, &ncols, &one, copya.Matrix(lda, opts), &lda, work.Matrix(ldwork, opts), &ldwork, &zero, b.Matrix(ldb, opts), &ldb)
									golapack.Dlacpy('F', &nrows, &nrhs, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb)

									//                             Solve LS or overdetermined system
									if m > 0 && n > 0 {
										golapack.Dlacpy('F', &m, &n, copya.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
										golapack.Dlacpy('F', &nrows, &nrhs, copyb.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb)
									}
									*srnamt = "DGELS "
									golapack.Dgels(trans, &m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork, &info)
									if info != 0 {
										Alaerh(path, []byte("DGELS "), &info, func() *int { y := 0; return &y }(), []byte{trans}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
									}

									//                             Check correctness of results
									ldwork = maxint(1, nrows)
									if nrows > 0 && nrhs > 0 {
										golapack.Dlacpy('F', &nrows, &nrhs, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), &ldb)
									}
									Dqrt16(trans, &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), &ldb, work, result.GetPtr(0))

									if (itran == 1 && m >= n) || (itran == 2 && m < n) {
										//                                Solving LS system
										result.Set(1, Dqrt17(trans, func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), work, &lwork))
									} else {
										//                                Solving overdetermined system
										result.Set(1, Dqrt14(trans, &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork))
									}

									//                             Print information about the tests that
									//                             did not pass the threshold.
									for k = 1; k <= 2; k++ {
										if result.Get(k-1) >= (*thresh) {
											if nfail == 0 && nerrs == 0 {
												Alahd(path)
											}
											t.Fail()
											fmt.Printf(" TRANS='%c', M=%5d, N=%5d, NRHS=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, nb, itype, k, result.Get(k-1))
											nfail = nfail + 1
										}
									}
									nrun = nrun + 2
								}
							}

							//                       Test DGETSLS
							//
							//                       Generate a matrix of scaling type ISCALE
							Dqrt13(&iscale, &m, &n, copya.Matrix(lda, opts), &lda, &norma, &iseed)
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
											trans = 'T'
											nrows = n
											ncols = m
										}
										ldwork = maxint(1, ncols)

										//                             Set up a consistent rhs
										if ncols > 0 {
											golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(ncols*nrhs), work)
											goblas.Dscal(toPtr(ncols*nrhs), toPtrf64(one/float64(ncols)), work, toPtr(1))
										}
										goblas.Dgemm(mat.TransByte(trans), NoTrans, &nrows, &nrhs, &ncols, &one, copya.Matrix(lda, opts), &lda, work.Matrix(ldwork, opts), &ldwork, &zero, b.Matrix(ldb, opts), &ldb)
										golapack.Dlacpy('F', &nrows, &nrhs, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb)

										//                             Solve LS or overdetermined system
										if m > 0 && n > 0 {
											golapack.Dlacpy('F', &m, &n, copya.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
											golapack.Dlacpy('F', &nrows, &nrhs, copyb.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb)
										}
										*srnamt = "DGETSLS "
										golapack.Dgetsls(trans, &m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork, &info)
										if info != 0 {
											Alaerh(path, []byte("DGETSLS "), &info, func() *int { y := 0; return &y }(), []byte{trans}, &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
										}

										//                             Check correctness of results
										ldwork = maxint(1, nrows)
										if nrows > 0 && nrhs > 0 {
											golapack.Dlacpy('F', &nrows, &nrhs, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), &ldb)
										}
										Dqrt16(trans, &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), &ldb, work, result.GetPtr(14))

										if (itran == 1 && m >= n) || (itran == 2 && m < n) {
											//                                Solving LS system
											result.Set(15, Dqrt17(trans, func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), work, &lwork))
										} else {
											//                                Solving overdetermined system
											result.Set(15, Dqrt14(trans, &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork))
										}

										//                             Print information about the tests that
										//                             did not pass the threshold.
										for k = 15; k <= 16; k++ {
											if result.Get(k-1) >= (*thresh) {
												if nfail == 0 && nerrs == 0 {
													Alahd(path)
												}
												t.Fail()
												fmt.Printf(" TRANS='%c M=%5d, N=%5d, NRHS=%4d, MB=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, mb, nb, itype, k, result.Get(k-1))
												nfail = nfail + 1
											}
										}
										nrun = nrun + 2
									}
								}
							}
						}

						//                    Generate a matrix of scaling type ISCALE and rank
						//                    type IRANK.
						Dqrt15(&iscale, &irank, &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, copyb.Matrix(ldb, opts), &ldb, copys, &rank, &norma, &normb, &iseed, work, &lwork)

						//                    workspace used: MAX(M+MIN(M,N),NRHS*MIN(M,N),2*N+M)
						ldwork = maxint(1, m)

						//                    Loop for testing different block sizes.
						for inb = 1; inb <= (*nnb); inb++ {
							nb = (*nbval)[inb-1]
							Xlaenv(1, nb)
							Xlaenv(3, (*nxval)[inb-1])

							//                       Test DGELSY
							//
							//                       DGELSY:  Compute the minimum-norm solution X
							//                       to min( norm( A * X - B ) )
							//                       using the rank-revealing orthogonal
							//                       factorization.
							//
							//                       Initialize vector iwork.
							for j = 1; j <= n; j++ {
								iwork[j-1] = 0
							}

							golapack.Dlacpy('F', &m, &n, copya.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb)
							*srnamt = "DGELSY"
							golapack.Dgelsy(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, &iwork, &rcond, &crank, work, &lwlsy, &info)
							if info != 0 {
								Alaerh(path, []byte("DGELSY"), &info, func() *int { y := 0; return &y }(), []byte(" "), &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       Test 3:  Compute relative error in svd
							//                                workspace: M*N + 4*MIN(M,N) + MAX(M,N)
							result.Set(2, Dqrt12(&crank, &crank, a.Matrix(lda, opts), &lda, copys, work, &lwork))

							//                       Test 4:  Compute error in solution
							//                                workspace:  M*NRHS + M
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork)
							Dqrt16('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork, work.Off(m*nrhs+1-1), result.GetPtr(3))

							//                       Test 5:  Check norm of r'*A
							//                                workspace: NRHS*(M+N)
							result.Set(4, zero)
							if m > crank {
								result.Set(4, Dqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), work, &lwork))
							}

							//                       Test 6:  Check if x is in the rowspace of A
							//                                workspace: (M+NRHS)*(N+2)
							result.Set(5, zero)

							if n > crank {
								result.Set(5, Dqrt14('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Test DGELSS
							//
							//                       DGELSS:  Compute the minimum-norm solution X
							//                       to min( norm( A * X - B ) )
							//                       using the SVD.
							golapack.Dlacpy('F', &m, &n, copya.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb)
							*srnamt = "DGELSS"
							golapack.Dgelss(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, s, &rcond, &crank, work, &lwork, &info)
							if info != 0 {
								Alaerh(path, []byte("DGELSS"), &info, func() *int { y := 0; return &y }(), []byte(" "), &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       workspace used: 3*min(m,n) +
							//                                       max(2*min(m,n),nrhs,max(m,n))
							//
							//                       Test 7:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(&mnmin, toPtrf64(-one), copys, toPtr(1), s, toPtr(1))
								result.Set(6, goblas.Dasum(&mnmin, s, toPtr(1))/goblas.Dasum(&mnmin, copys, toPtr(1))/(eps*float64(mnmin)))
							} else {
								result.Set(6, zero)
							}

							//                       Test 8:  Compute error in solution
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork)
							Dqrt16('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork, work.Off(m*nrhs+1-1), result.GetPtr(7))

							//                       Test 9:  Check norm of r'*A
							result.Set(8, zero)
							if m > crank {
								result.Set(8, Dqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), work, &lwork))
							}

							//                       Test 10:  Check if x is in the rowspace of A
							result.Set(9, zero)
							if n > crank {
								result.Set(9, Dqrt14('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Test DGELSD
							//
							//                       DGELSD:  Compute the minimum-norm solution X
							//                       to min( norm( A * X - B ) ) using a
							//                       divide and conquer SVD.
							//
							//                       Initialize vector iwork.
							for j = 1; j <= n; j++ {
								iwork[j-1] = 0
							}

							golapack.Dlacpy('F', &m, &n, copya.Matrix(lda, opts), &lda, a.Matrix(lda, opts), &lda)
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, b.Matrix(ldb, opts), &ldb)

							*srnamt = "DGELSD"
							golapack.Dgelsd(&m, &n, &nrhs, a.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, s, &rcond, &crank, work, &lwork, &iwork, &info)
							if info != 0 {
								Alaerh(path, []byte("DGELSD"), &info, func() *int { y := 0; return &y }(), []byte(" "), &m, &n, &nrhs, toPtr(-1), &nb, &itype, &nfail, &nerrs)
							}

							//                       Test 11:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(&mnmin, toPtrf64(-one), copys, toPtr(1), s, toPtr(1))
								result.Set(10, goblas.Dasum(&mnmin, s, toPtr(1))/goblas.Dasum(&mnmin, copys, toPtr(1))/(eps*float64(mnmin)))
							} else {
								result.Set(10, zero)
							}

							//                       Test 12:  Compute error in solution
							golapack.Dlacpy('F', &m, &nrhs, copyb.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork)
							Dqrt16('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work.Matrix(ldwork, opts), &ldwork, work.Off(m*nrhs+1-1), result.GetPtr(11))

							//                       Test 13:  Check norm of r'*A
							result.Set(12, zero)
							if m > crank {
								result.Set(12, Dqrt17('N', func() *int { y := 1; return &y }(), &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, copyb.Matrix(ldb, opts), &ldb, c.Matrix(ldb, opts), work, &lwork))
							}

							//                       Test 14:  Check if x is in the rowspace of A
							result.Set(13, zero)
							if n > crank {
								result.Set(13, Dqrt14('N', &m, &n, &nrhs, copya.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), &ldb, work, &lwork))
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 14; k++ {
								if result.Get(k-1) >= (*thresh) {
									if nfail == 0 && nerrs == 0 {
										Alahd(path)
									}
									t.Fail()
									fmt.Printf(" M=%5d, N=%5d, NRHS=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", m, n, nrhs, nb, itype, k, result.Get(k-1))
									nfail = nfail + 1
								}
							}
							nrun = nrun + 12

						}
					label110:
					}
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 105840
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasvm(path, &nfail, &nrun, &nerrs)
}
