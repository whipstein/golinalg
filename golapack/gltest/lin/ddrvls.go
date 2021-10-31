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

// ddrvls tests the least squares driver routines DGELS, DGETSLS, DGELSS, DGELSY,
// and DGELSD.
func ddrvls(dotype []bool, nm int, mval []int, nn int, nval []int, nns int, nsval []int, nnb int, nbval []int, nxval []int, thresh float64, tsterr bool, a, copya, b, copyb, c, s, copys *mat.Vector, t *testing.T) {
	var trans mat.MatTrans
	var eps, one, rcond, zero float64
	var crank, i, im, imb, in, inb, info, ins, irank, iscale, itype, j, k, lda, ldb, ldwork, liwork, lwlsy, lwork, lworkDgels, lworkDgelsd, lworkDgelss, lworkDgelsy, lworkDgetsls, m, mb, mmax, mnmin, n, nb, ncols, nerrs, nfail, nmax, nrhs, nrows, nrun, nsmax, rank, smlsiz int
	var err error

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
	path := "Dls"
	alasvmStart(path)
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
	xlaenv(2, 2)
	xlaenv(9, smlsiz)
	if tsterr {
		derrls(path, t)
	}

	//     Print the header if NM = 0 or NN = 0 and THRESH = 0.
	if (nm == 0 || nn == 0) && thresh == zero {
		alahd(path)
	}
	(*infot) = 0
	xlaenv(2, 2)
	xlaenv(9, smlsiz)

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
	wq := vf(1)

	//     Compute workspace needed for routines
	//     DQRT14, DQRT17 (two side cases), DQRT15 and DQRT12
	lwork = max(1, (m+n)*nrhs, (n+nrhs)*(m+2), (m+nrhs)*(n+2), max(m+mnmin, nrhs*mnmin, 2*n+m), max(m*n+4*mnmin+max(m, n), m*n+2*mnmin+4*n))
	liwork = 1

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

									//                             Compute workspace needed for DGELS
									if info, err = golapack.Dgels(trans, m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), wq, -1); err != nil {
										panic(err)
									}
									lworkDgels = int(wq.Get(0))
									//                             Compute workspace needed for DGETSLS
									if info, err = golapack.Dgetsls(trans, m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), wq, -1); err != nil {
										panic(err)
									}
									lworkDgetsls = int(wq.Get(0))
								}
							}
							//                       Compute workspace needed for DGELSY
							if crank, err = golapack.Dgelsy(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), &iwq, rcond, wq, -1); err != nil {
								panic(err)
							}
							lworkDgelsy = int(wq.Get(0))
							//                       Compute workspace needed for DGELSS
							if crank, info, err = golapack.Dgelss(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), s, rcond, wq, -1); err != nil {
								panic(err)
							}
							lworkDgelss = int(wq.Get(0))
							//                       Compute workspace needed for DGELSD
							if crank, info, err = golapack.Dgelsd(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), s, rcond, wq, -1, &iwq); err != nil {
								panic(err)
							}
							lworkDgelsd = int(wq.Get(0))
							//                       Compute LIWORK workspace needed for DGELSY and DGELSD
							liwork = max(liwork, n, iwq[0])
							//                       Compute LWORK workspace needed for all functions
							lwork = max(lwork, lworkDgels, lworkDgetsls, lworkDgelsy, lworkDgelss, lworkDgelsd)
						}
					}
				}
			}
		}
	}

	lwlsy = lwork

	work := vf(lwork)
	iwork := make([]int, liwork)

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
							goto label110
						}

						if irank == 1 {
							//                       Test DGELS
							//
							//                       Generate a matrix of scaling type ISCALE
							_, iseed = dqrt13(iscale, m, n, copya.Matrix(lda, opts), iseed)
							for inb = 1; inb <= nnb; inb++ {
								nb = nbval[inb-1]
								xlaenv(1, nb)
								xlaenv(3, nxval[inb-1])

								for _, trans = range mat.IterMatTrans(false) {
									if trans == NoTrans {
										nrows = m
										ncols = n
									} else {
										nrows = n
										ncols = m
									}
									ldwork = max(1, ncols)

									//                             Set up a consistent rhs
									if ncols > 0 {
										golapack.Dlarnv(2, &iseed, ncols*nrhs, work)
										goblas.Dscal(ncols*nrhs, one/float64(ncols), work.Off(0, 1))
									}
									if err = goblas.Dgemm(trans, NoTrans, nrows, nrhs, ncols, one, copya.Matrix(lda, opts), work.Matrix(ldwork, opts), zero, b.Matrix(ldb, opts)); err != nil {
										panic(err)
									}
									golapack.Dlacpy(Full, nrows, nrhs, b.Matrix(ldb, opts), copyb.Matrix(ldb, opts))

									//                             Solve LS or overdetermined system
									if m > 0 && n > 0 {
										golapack.Dlacpy(Full, m, n, copya.Matrix(lda, opts), a.Matrix(lda, opts))
										golapack.Dlacpy(Full, nrows, nrhs, copyb.Matrix(ldb, opts), b.Matrix(ldb, opts))
									}
									*srnamt = "Dgels"
									if info, err = golapack.Dgels(trans, m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork); info != 0 || err != nil {
										nerrs = alaerh(path, "Dgels", info, 0, []byte{trans.Byte()}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
									}

									//                             Check correctness of results
									ldwork = max(1, nrows)
									if nrows > 0 && nrhs > 0 {
										golapack.Dlacpy(Full, nrows, nrhs, copyb.Matrix(ldb, opts), c.Matrix(ldb, opts))
									}
									result.Set(0, dqrt16(trans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), c.Matrix(ldb, opts), work))

									if (trans == NoTrans && m >= n) || (trans == Trans && m < n) {
										//                                Solving LS system
										result.Set(1, dqrt17(trans, 1, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), copyb.Matrix(ldb, opts), c.Matrix(ldb, opts), work, lwork))
									} else {
										//                                Solving overdetermined system
										result.Set(1, dqrt14(trans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork))
									}

									//                             Print information about the tests that
									//                             did not pass the threshold.
									for k = 1; k <= 2; k++ {
										if result.Get(k-1) >= thresh {
											if nfail == 0 && nerrs == 0 {
												alahd(path)
											}
											t.Fail()
											fmt.Printf(" TRANS='%c', M=%5d, N=%5d, NRHS=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, nb, itype, k, result.Get(k-1))
											nfail++
										}
									}
									nrun += 2
								}
							}

							//                       Test DGETSLS
							//
							//                       Generate a matrix of scaling type ISCALE
							_, iseed = dqrt13(iscale, m, n, copya.Matrix(lda, opts), iseed)
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
											nrows = n
											ncols = m
										}
										ldwork = max(1, ncols)

										//                             Set up a consistent rhs
										if ncols > 0 {
											golapack.Dlarnv(2, &iseed, ncols*nrhs, work)
											goblas.Dscal(ncols*nrhs, one/float64(ncols), work.Off(0, 1))
										}
										if err = goblas.Dgemm(trans, NoTrans, nrows, nrhs, ncols, one, copya.Matrix(lda, opts), work.Matrix(ldwork, opts), zero, b.Matrix(ldb, opts)); err != nil {
											panic(err)
										}
										golapack.Dlacpy(Full, nrows, nrhs, b.Matrix(ldb, opts), copyb.Matrix(ldb, opts))

										//                             Solve LS or overdetermined system
										if m > 0 && n > 0 {
											golapack.Dlacpy(Full, m, n, copya.Matrix(lda, opts), a.Matrix(lda, opts))
											golapack.Dlacpy(Full, nrows, nrhs, copyb.Matrix(ldb, opts), b.Matrix(ldb, opts))
										}
										*srnamt = "Dgetsls"
										if info, err = golapack.Dgetsls(trans, m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork); info != 0 || err != nil {
											nerrs = alaerh(path, "Dgetsls", info, 0, []byte{trans.Byte()}, m, n, nrhs, -1, nb, itype, nfail, nerrs)
										}

										//                             Check correctness of results
										ldwork = max(1, nrows)
										if nrows > 0 && nrhs > 0 {
											golapack.Dlacpy(Full, nrows, nrhs, copyb.Matrix(ldb, opts), c.Matrix(ldb, opts))
										}
										result.Set(14, dqrt16(trans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), c.Matrix(ldb, opts), work))

										if (trans == NoTrans && m >= n) || (trans == Trans && m < n) {
											//                                Solving LS system
											result.Set(15, dqrt17(trans, 1, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), copyb.Matrix(ldb, opts), c.Matrix(ldb, opts), work, lwork))
										} else {
											//                                Solving overdetermined system
											result.Set(15, dqrt14(trans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork))
										}

										//                             Print information about the tests that
										//                             did not pass the threshold.
										for k = 15; k <= 16; k++ {
											if result.Get(k-1) >= thresh {
												if nfail == 0 && nerrs == 0 {
													alahd(path)
												}
												t.Fail()
												fmt.Printf(" TRANS=%s M=%5d, N=%5d, NRHS=%4d, MB=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", trans, m, n, nrhs, mb, nb, itype, k, result.Get(k-1))
												nfail++
											}
										}
										nrun += 2
									}
								}
							}
						}

						//                    Generate a matrix of scaling type ISCALE and rank
						//                    type IRANK.
						rank, _, _, iseed = dqrt15(iscale, irank, m, n, nrhs, copya.Matrix(lda, opts), copyb.Matrix(ldb, opts), copys, iseed, work, lwork)

						//                    workspace used: MAX(M+MIN(M,N),NRHS*MIN(M,N),2*N+M)
						ldwork = max(1, m)

						//                    Loop for testing different block sizes.
						for inb = 1; inb <= nnb; inb++ {
							nb = nbval[inb-1]
							xlaenv(1, nb)
							xlaenv(3, nxval[inb-1])

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

							golapack.Dlacpy(Full, m, n, copya.Matrix(lda, opts), a.Matrix(lda, opts))
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), b.Matrix(ldb, opts))
							*srnamt = "Dgelsy"
							if crank, err = golapack.Dgelsy(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), &iwork, rcond, work, lwlsy); err != nil {
								nerrs = alaerh(path, "Dgelsy", info, 0, []byte(" "), m, n, nrhs, -1, nb, itype, nfail, nerrs)
							}

							//                       Test 3:  Compute relative error in svd
							//                                workspace: M*N + 4*MIN(M,N) + MAX(M,N)
							result.Set(2, dqrt12(crank, crank, a.Matrix(lda, opts), copys, work, lwork))

							//                       Test 4:  Compute error in solution
							//                                workspace:  M*NRHS + M
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), work.Matrix(ldwork, opts))
							result.Set(3, dqrt16(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work.Matrix(ldwork, opts), work.Off(m*nrhs)))

							//                       Test 5:  Check norm of r'*A
							//                                workspace: NRHS*(M+N)
							result.Set(4, zero)
							if m > crank {
								result.Set(4, dqrt17(NoTrans, 1, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), copyb.Matrix(ldb, opts), c.Matrix(ldb, opts), work, lwork))
							}

							//                       Test 6:  Check if x is in the rowspace of A
							//                                workspace: (M+NRHS)*(N+2)
							result.Set(5, zero)

							if n > crank {
								result.Set(5, dqrt14(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork))
							}

							//                       Test DGELSS
							//
							//                       DGELSS:  Compute the minimum-norm solution X
							//                       to min( norm( A * X - B ) )
							//                       using the SVD.
							golapack.Dlacpy(Full, m, n, copya.Matrix(lda, opts), a.Matrix(lda, opts))
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), b.Matrix(ldb, opts))
							*srnamt = "Dgelss"
							if crank, info, err = golapack.Dgelss(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), s, rcond, work, lwork); info != 0 || err != nil {
								nerrs = alaerh(path, "Dgelss", info, 0, []byte(" "), m, n, nrhs, -1, nb, itype, nfail, nerrs)
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
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), work.Matrix(ldwork, opts))
							result.Set(7, dqrt16(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work.Matrix(ldwork, opts), work.Off(m*nrhs)))

							//                       Test 9:  Check norm of r'*A
							result.Set(8, zero)
							if m > crank {
								result.Set(8, dqrt17(NoTrans, 1, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), copyb.Matrix(ldb, opts), c.Matrix(ldb, opts), work, lwork))
							}

							//                       Test 10:  Check if x is in the rowspace of A
							result.Set(9, zero)
							if n > crank {
								result.Set(9, dqrt14(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork))
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

							golapack.Dlacpy(Full, m, n, copya.Matrix(lda, opts), a.Matrix(lda, opts))
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), b.Matrix(ldb, opts))

							*srnamt = "Dgelsd"
							if crank, info, err = golapack.Dgelsd(m, n, nrhs, a.Matrix(lda, opts), b.Matrix(ldb, opts), s, rcond, work, lwork, &iwork); info != 0 || err != nil {
								nerrs = alaerh(path, "Dgelsd", info, 0, []byte(" "), m, n, nrhs, -1, nb, itype, nfail, nerrs)
							}

							//                       Test 11:  Compute relative error in svd
							if rank > 0 {
								goblas.Daxpy(mnmin, -one, copys.Off(0, 1), s.Off(0, 1))
								result.Set(10, goblas.Dasum(mnmin, s.Off(0, 1))/goblas.Dasum(mnmin, copys.Off(0, 1))/(eps*float64(mnmin)))
							} else {
								result.Set(10, zero)
							}

							//                       Test 12:  Compute error in solution
							golapack.Dlacpy(Full, m, nrhs, copyb.Matrix(ldb, opts), work.Matrix(ldwork, opts))
							result.Set(11, dqrt16(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work.Matrix(ldwork, opts), work.Off(m*nrhs)))

							//                       Test 13:  Check norm of r'*A
							result.Set(12, zero)
							if m > crank {
								result.Set(12, dqrt17(NoTrans, 1, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), copyb.Matrix(ldb, opts), c.Matrix(ldb, opts), work, lwork))
							}

							//                       Test 14:  Check if x is in the rowspace of A
							result.Set(13, zero)
							if n > crank {
								result.Set(13, dqrt14(NoTrans, m, n, nrhs, copya.Matrix(lda, opts), b.Matrix(ldb, opts), work, lwork))
							}

							//                       Print information about the tests that did not
							//                       pass the threshold.
							for k = 3; k <= 14; k++ {
								if result.Get(k-1) >= thresh {
									if nfail == 0 && nerrs == 0 {
										alahd(path)
									}
									t.Fail()
									fmt.Printf(" M=%5d, N=%5d, NRHS=%4d, NB=%4d, type%2d, test(%2d)=%12.5f\n", m, n, nrhs, nb, itype, k, result.Get(k-1))
									nfail++
								}
							}
							nrun += 12

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
	// alasvm(path, nfail, nrun, nerrs)
	alasvmEnd(nfail, nrun, nerrs)
}
