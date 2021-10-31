package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkqr tests DGEQRF, DORGQR and DORMQR.
func dchkqr(dotype []bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, nxval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, af, aq, ar, ac, b, x, xact, tau, work, rwork *mat.Vector, iwork []int, t *testing.T) {
	var dist, _type byte
	var anorm, cndnum, zero float64
	var i, ik, im, imat, in, inb, info, k, kl, ku, lda, lwork, m, minmn, mode, n, nb, nerrs, nfail, nk, nrun, nt, ntests, ntypes, nx int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kval := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	ntests = 9
	ntypes = 8
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Dqr"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		derrqr(path, t)
	}
	(*infot) = 0
	xlaenv(2, 2)

	lda = nmax
	lwork = nmax * max(nmax, nrhs)

	//     Do for each value of M in MVAL.
	for im = 1; im <= nm; im++ {
		m = mval[im-1]

		//        Do for each value of N in NVAL.
		for in = 1; in <= nn; in++ {
			n = nval[in-1]
			minmn = min(m, n)
			for imat = 1; imat <= ntypes; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !dotype[imat-1] {
					goto label50
				}

				//              Set up parameters with DLATB4 and generate a test matrix
				//              with DLATMS.
				_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, m, n)

				*srnamt = "Dlatms"
				if info, _ = matgen.Dlatms(m, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a.Matrix(lda, opts), work); info != 0 {
					nerrs = alaerh(path, "Dlatms", info, 0, []byte(" "), m, n, -1, -1, -1, imat, nfail, nerrs)
					goto label50
				}

				//              Set some values for K: the first value must be MINMN,
				//              corresponding to the call of DQRT01; other values are
				//              used in the calls of DQRT02, and must not exceed MINMN.
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
					for inb = 1; inb <= nnb; inb++ {
						nb = nbval[inb-1]
						xlaenv(1, nb)
						nx = nxval[inb-1]
						xlaenv(3, nx)
						for i = 1; i <= ntests; i++ {
							result.Set(i-1, zero)
						}
						nt = 2
						if ik == 1 {
							//                       Test DGEQRF
							dqrt01(m, n, a.Matrix(lda, opts), af.Matrix(lda, opts), aq.Matrix(lda, opts), ar.Matrix(lda, opts), tau, work, lwork, rwork, result)

							//                       Test DGEQRFP
							dqrt01p(m, n, a.Matrix(lda, opts), af.Matrix(lda, opts), aq.Matrix(lda, opts), ar.Matrix(lda, opts), tau, work, lwork, rwork, result.Off(7))
							if !dgennd(m, n, af.Matrix(lda, opts)) {
								result.Set(8, 2*thresh)
							}
							nt++
						} else if m >= n {
							//                       Test DORGQR, using factorization
							//                       returned by DQRT01
							dqrt02(m, n, k, a.Matrix(lda, opts), af.Matrix(lda, opts), aq.Matrix(lda, opts), ar.Matrix(lda, opts), tau, work, lwork, rwork, result)
						}
						if m >= k {
							//                       Test DORMQR, using factorization returned
							//                       by DQRT01
							dqrt03(m, n, k, af.Matrix(lda, opts), ac.Matrix(lda, opts), ar.Matrix(lda, opts), aq.Matrix(lda, opts), tau, work, lwork, rwork, result.Off(2))
							nt += 4

							//                       If M>=N and K=N, call DGEQRS to solve a system
							//                       with NRHS right hand sides and compute the
							//                       residual.
							if k == n && inb == 1 {
								//                          Generate a solution and set the right
								//                          hand side.
								*srnamt = "Dlarhs"
								if err = Dlarhs(path, 'N', Full, NoTrans, m, n, 0, 0, nrhs, a.Matrix(lda, opts), xact.Matrix(lda, opts), b.Matrix(lda, opts), &iseed); err != nil {
									panic(err)
								}

								golapack.Dlacpy(Full, m, nrhs, b.Matrix(lda, opts), x.Matrix(lda, opts))
								*srnamt = "Dgeqrs"
								if err = dgeqrs(m, n, nrhs, af.Matrix(lda, opts), tau, x.Matrix(lda, opts), work, lwork); err != nil {
									nerrs = alaerh(path, "Dgeqrs", info, 0, []byte(" "), m, n, nrhs, -1, nb, imat, nfail, nerrs)
								}

								result.Set(6, dget02(NoTrans, m, n, nrhs, a.Matrix(lda, opts), x.Matrix(lda, opts), b.Matrix(lda, opts), rwork))
								nt++
							}
						}

						//                    Print information about the tests that did not
						//                    pass the threshold.
						for i = 1; i <= ntests; i++ {
							if result.Get(i-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								t.Fail()
								fmt.Printf(" M=%5d, N=%5d, K=%5d, NB=%4d, NX=%5d, _type %2d, test(%2d)=%12.5f\n", m, n, k, nb, nx, imat, i, result.Get(i-1))
								nfail++
							}
						}
						nrun = nrun + ntests
					}
				}
			label50:
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 42840
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
