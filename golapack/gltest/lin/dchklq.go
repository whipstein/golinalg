package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Dchklq tests DGELQF, DORGLQ and DORMLQ.
func Dchklq(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval, nxval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, af, aq, al, ac, b, x, xact, tau, work, rwork *mat.Vector, nout *int, t *testing.T) {
	var dist, _type byte
	var anorm, cndnum, zero float64
	var i, ik, im, imat, in, inb, info, k, kl, ku, lda, lwork, m, minmn, mode, n, nb, nerrs, nfail, nk, nrun, nt, ntests, ntypes, nx int

	result := vf(7)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kval := make([]int, 4)

	ntests = 7
	ntypes = 8
	zero = 0.0

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := []byte("DLQ")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrlq(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

	lda = (*nmax)
	lwork = (*nmax) * maxint(*nmax, *nrhs)

	//     Do for each value of M in MVAL.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]

		//        Do for each value of N in NVAL.
		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
			minmn = minint(m, n)
			for imat = 1; imat <= ntypes; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !(*dotype)[imat-1] {
					goto label50
				}

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				Dlatb4(path, &imat, &m, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "DLATMS"
				matgen.Dlatms(&m, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'N', a.Matrix(lda, opts), &lda, work, &info)

				//              Check error code from DLATMS.
				if info != 0 {
					Alaerh(path, []byte("DLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label50
				}

				//              Set some values for K: the first value must be MINMN,
				//              corresponding to the call of DLQT01; other values are
				//              used in the calls of DLQT02, and must not exceed MINMN.
				kval[0] = minmn
				kval[1] = 0
				kval[2] = 1
				kval[3] = minmn / 2
				if minmn == 0 {
					nk = 1
				} else if minmn == 1 {
					nk = 2
				} else if minmn <= 3 {
					nk = 3
				} else {
					nk = 4
				}

				//              Do for each value of K in KVAL
				for ik = 1; ik <= nk; ik++ {
					k = kval[ik-1]

					//                 Do for each pair of values (NB,NX) in NBVAL and NXVAL.
					for inb = 1; inb <= (*nnb); inb++ {
						nb = (*nbval)[inb-1]
						Xlaenv(1, nb)
						nx = (*nxval)[inb-1]
						Xlaenv(3, nx)
						for i = 1; i <= ntests; i++ {
							result.Set(i-1, zero)
						}
						nt = 2
						if ik == 1 {
							//                       Test DGELQF
							Dlqt01(&m, &n, a.Matrix(lda, opts), af.Matrix(lda, opts), aq.Matrix(lda, opts), al.Matrix(lda, opts), &lda, tau, work, &lwork, rwork, result)
						} else if m <= n {
							//                       Test DORGLQ, using factorization
							//                       returned by DLQT01
							Dlqt02(&m, &n, &k, a.Matrix(lda, opts), af.Matrix(lda, opts), aq.Matrix(lda, opts), al.Matrix(lda, opts), &lda, tau, work, &lwork, rwork, result)
						} else {
							result.Set(0, zero)
							result.Set(1, zero)
						}
						if m >= k {
							//                       Test DORMLQ, using factorization returned
							//                       by DLQT01
							Dlqt03(&m, &n, &k, af.Matrix(lda, opts), ac.Matrix(lda, opts), al.Matrix(lda, opts), aq.Matrix(lda, opts), &lda, tau, work, &lwork, rwork, result.Off(2))
							nt = nt + 4

							//                       If M>=N and K=N, call DGELQS to solve a system
							//                       with NRHS right hand sides and compute the
							//                       residual.
							if k == m && inb == 1 {
								//                          Generate a solution and set the right
								//                          hand side.
								*srnamt = "DLARHS"
								Dlarhs(path, toPtrByte('N'), 'F', 'N', &m, &n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), nrhs, a.Matrix(lda, opts), &lda, xact.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, &iseed, &info)

								golapack.Dlacpy('F', &m, nrhs, b.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda)
								*srnamt = "DGELQS"
								Dgelqs(&m, &n, nrhs, af.Matrix(lda, opts), &lda, tau, x.Matrix(lda, opts), &lda, work, &lwork, &info)

								//                          Check error code from DGELQS.
								if info != 0 {
									Alaerh(path, []byte("DGELQS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, nrhs, toPtr(-1), &nb, &imat, &nfail, &nerrs)
								}

								Dget02('N', &m, &n, nrhs, a.Matrix(lda, opts), &lda, x.Matrix(lda, opts), &lda, b.Matrix(lda, opts), &lda, rwork, result.GetPtr(6))
								nt = nt + 1
							} else {
								result.Set(6, zero)
							}
						} else {
							result.Set(2, zero)
							result.Set(3, zero)
							result.Set(4, zero)
							result.Set(5, zero)
						}

						//                    Print information about the tests that did not
						//                    pass the threshold.
						for i = 1; i <= nt; i++ {
							if result.Get(i-1) >= (*thresh) {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" M=%5d, N=%5d, K=%5d, NB=%4d, NX=%5d, _type %2d, test(%2d)=%12.5f\n", m, n, k, nb, nx, imat, i, result.Get(i-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + nt
					}
				}
			label50:
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 28784
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
