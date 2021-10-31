package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkqr tests ZGEQRF, ZUNGQR and CUNMQR.
func zchkqr(dotype []bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, nxval []int, nrhs int, thresh float64, tsterr bool, nmax int, a, af, aq, ar, ac, b, x, xact, tau, work *mat.CVector, rwork *mat.Vector, iwork []int, t *testing.T) {
	var dist, _type byte
	var anorm, cndnum, zero float64
	var i, ik, im, imat, in, inb, info, k, kl, ku, lda, lwork, m, minmn, mode, n, nb, nerrs, nfail, nk, nrun, nt, ntests, ntypes, nx int
	var err error

	result := vf(9)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	kval := make([]int, 4)

	ntests = 9
	ntypes = 8
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Zqr"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrqr(path, t)
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

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with Zlatms.
				_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, m, n)

				*srnamt = "Zlatms"
				if err = matgen.Zlatms(m, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a.CMatrix(lda, opts), work); err != nil {
					t.Fail()
					nerrs = alaerh(path, "Zlatms", info, 0, []byte{' '}, m, n, -1, -1, -1, imat, nfail, nerrs)
					goto label50
				}

				//              Set some values for K: the first value must be MINMN,
				//              corresponding to the call of ZQRT01; other values are
				//              used in the calls of ZQRT02, and must not exceed MINMN.
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

					//                 Do for each pair of values (nb,nx) in NBVAL and NXVAL.
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
							//                       Test ZGEQRF
							zqrt01(m, n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), ar.CMatrix(lda, opts), tau, work, lwork, rwork, result.Off(0))

							//                       Test ZGEQRFP
							zqrt01p(m, n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), ar.CMatrix(lda, opts), tau, work, lwork, rwork, result.Off(7))
							if !zgennd(m, n, af.CMatrix(lda, opts)) {
								result.Set(8, 2*thresh)
							}
							nt = nt + 1
						} else if m >= n {
							//                       Test ZUNGQR, using factorization
							//                       returned by ZQRT01
							zqrt02(m, n, k, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), ar.CMatrix(lda, opts), tau, work, lwork, rwork, result.Off(0))
						}
						if m >= k {
							//                       Test ZUNMQR, using factorization returned
							//                       by ZQRT01
							zqrt03(m, n, k, af.CMatrix(lda, opts), ac.CMatrix(lda, opts), ar.CMatrix(lda, opts), aq.CMatrix(lda, opts), tau, work, lwork, rwork, result.Off(2))
							nt = nt + 4

							//                       If M>=N and k=N, call Zgeqrs to solve a system
							//                       with NRHS right hand sides and compute the
							//                       residual.
							if k == n && inb == 1 {
								//                          Generate a solution and set the right
								//                          hand side.
								*srnamt = "zlarhs"
								if err = zlarhs(path, 'N', Full, NoTrans, m, n, 0, 0, nrhs, a.CMatrix(lda, opts), xact.CMatrix(lda, opts), b.CMatrix(lda, opts), &iseed); err != nil {
									panic(err)
								}

								golapack.Zlacpy(Full, m, nrhs, b.CMatrix(lda, opts), x.CMatrix(lda, opts))
								*srnamt = "zgeqrs"
								if err = zgeqrs(m, n, nrhs, af.CMatrix(lda, opts), tau, x.CMatrix(lda, opts), work, lwork); err != nil {
									t.Fail()
									nerrs = alaerh(path, "zgeqrs", info, 0, []byte{' '}, m, n, nrhs, -1, nb, imat, nfail, nerrs)
								}

								*result.GetPtr(6) = zget02(NoTrans, m, n, nrhs, a.CMatrix(lda, opts), x.CMatrix(lda, opts), b.CMatrix(lda, opts), rwork)
								nt = nt + 1
							}
						}

						//                    Print information about the tests that did not
						//                    pass the threshold.
						for i = 1; i <= ntests; i++ {
							if result.Get(i-1) >= thresh {
								t.Fail()
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								fmt.Printf(" m=%5d, n=%5d, k=%5d, nb=%4d, nx=%5d, _type %2d, test(%2d)=%12.5f\n", m, n, k, nb, nx, imat, i, result.Get(i-1))
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

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
