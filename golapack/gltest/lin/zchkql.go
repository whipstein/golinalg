package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkql tests ZGEQLF, ZUNGQL and CUNMQL.
func Zchkql(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nxval *[]int, nrhs *int, thresh *float64, tsterr *bool, nmax *int, a, af, aq, al, ac, b, x, xact, tau, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
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
	path := []byte("ZQL")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Zerrql(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

	lda = (*nmax)
	lwork = (*nmax) * max(*nmax, *nrhs)

	//     Do for each value of M in MVAL.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]

		//        Do for each value of N in NVAL.
		for in = 1; in <= (*nn); in++ {
			n = (*nval)[in-1]
			minmn = min(m, n)
			for imat = 1; imat <= ntypes; imat++ {
				//              Do the tests only if DOTYPE( IMAT ) is true.
				if !(*dotype)[imat-1] {
					goto label50
				}

				//              Set up parameters with ZLATB4 and generate a test matrix
				//              with ZLATMS.
				Zlatb4(path, &imat, &m, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

				*srnamt = "ZLATMS"
				matgen.Zlatms(&m, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, 'N', a.CMatrix(lda, opts), &lda, work, &info)

				//              Check error code from ZLATMS.
				if info != 0 {
					t.Fail()
					Alaerh(path, []byte("ZLATMS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
					goto label50
				}

				//              Set some values for K: the first value must be MINMN,
				//              corresponding to the call of ZQLT01; other values are
				//              used in the calls of ZQLT02, and must not exceed MINMN.
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
							//                       Test ZGEQLF
							Zqlt01(&m, &n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), al.CMatrix(lda, opts), &lda, tau, work, &lwork, rwork, result.Off(0))
						} else if m >= n {
							//                       Test ZUNGQL, using factorization
							//                       returned by ZQLT01
							Zqlt02(&m, &n, &k, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), al.CMatrix(lda, opts), &lda, tau, work, &lwork, rwork, result.Off(0))
						}
						if m >= k {
							//                       Test ZUNMQL, using factorization returned
							//                       by ZQLT01
							Zqlt03(&m, &n, &k, af.CMatrix(lda, opts), ac.CMatrix(lda, opts), al.CMatrix(lda, opts), aq.CMatrix(lda, opts), &lda, tau, work, &lwork, rwork, result.Off(2))
							nt = nt + 4

							//                       If M>=N and K=N, call ZGEQLS to solve a system
							//                       with NRHS right hand sides and compute the
							//                       residual.
							if k == n && inb == 1 {
								//                          Generate a solution and set the right
								//                          hand side.
								*srnamt = "ZLARHS"
								Zlarhs(path, 'N', 'F', 'N', &m, &n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), nrhs, a.CMatrix(lda, opts), &lda, xact.CMatrix(lda, opts), &lda, b.CMatrix(lda, opts), &lda, &iseed, &info)

								golapack.Zlacpy('F', &m, nrhs, b.CMatrix(lda, opts), &lda, x.CMatrix(lda, opts), &lda)
								*srnamt = "ZGEQLS"
								Zgeqls(&m, &n, nrhs, af.CMatrix(lda, opts), &lda, tau, x.CMatrix(lda, opts), &lda, work, &lwork, &info)

								//                          Check error code from ZGEQLS.
								if info != 0 {
									t.Fail()
									Alaerh(path, []byte("ZGEQLS"), &info, func() *int { y := 0; return &y }(), []byte{' '}, &m, &n, nrhs, toPtr(-1), &nb, &imat, &nfail, &nerrs)
								}

								Zget02('N', &m, &n, nrhs, a.CMatrix(lda, opts), &lda, x.CMatrixOff(m-n, lda, opts), &lda, b.CMatrix(lda, opts), &lda, rwork, result.GetPtr(6))
								nt = nt + 1
							}
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

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
