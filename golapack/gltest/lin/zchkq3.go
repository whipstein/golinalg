package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkq3 tests ZGEQP3.
func Zchkq3(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nxval *[]int, thresh *float64, a, copya *mat.CVector, s *mat.Vector, tau, work *mat.CVector, rwork *mat.Vector, iwork *[]int, nout *int, t *testing.T) {
	var czero complex128
	var eps, one, zero float64
	var i, ihigh, ilow, im, imode, in, inb, info, istep, k, lda, lw, lwork, m, mnmin, mode, n, nb, nerrs, nfail, nrun, ntests, ntypes, nx int

	result := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntypes = 6
	ntests = 3
	one = 1.0
	zero = 0.0
	czero = (0.0 + 0.0*1i)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := []byte("ZQ3")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	eps = golapack.Dlamch(Epsilon)
	(*infot) = 0

	for im = 1; im <= (*nm); im++ {
		//        Do for each value of M in MVAL.
		m = (*mval)[im-1]
		lda = maxint(1, m)

		for in = 1; in <= (*nn); in++ {
			//           Do for each value of N in NVAL.
			n = (*nval)[in-1]
			mnmin = minint(m, n)
			lwork = maxint(1, m*maxint(m, n)+4*mnmin+maxint(m, n))
			//
			for imode = 1; imode <= ntypes; imode++ {
				if !(*dotype)[imode-1] {
					goto label70
				}

				//              Do for each _type of matrix
				//                 1:  zero matrix
				//                 2:  one small singular value
				//                 3:  geometric distribution of singular values
				//                 4:  first n/2 columns fixed
				//                 5:  last n/2 columns fixed
				//                 6:  every second column fixed
				mode = imode
				if imode > 3 {
					mode = 1
				}

				//              Generate test matrix of size m by n using
				//              singular value distribution indicated by `mode'.
				for i = 1; i <= n; i++ {
					(*iwork)[i-1] = 0
				}
				if imode == 1 {
					golapack.Zlaset('F', &m, &n, &czero, &czero, copya.CMatrix(lda, opts), &lda)
					for i = 1; i <= mnmin; i++ {
						s.Set(i-1, zero)
					}
				} else {
					matgen.Zlatms(&m, &n, 'U', &iseed, 'N', s, &mode, toPtrf64(one/eps), &one, &m, &n, 'N', copya.CMatrix(lda, opts), &lda, work, &info)
					if imode >= 4 {
						if imode == 4 {
							ilow = 1
							istep = 1
							ihigh = maxint(1, n/2)
						} else if imode == 5 {
							ilow = maxint(1, n/2)
							istep = 1
							ihigh = n
						} else if imode == 6 {
							ilow = 1
							istep = 2
							ihigh = n
						}
						for i = ilow; i <= ihigh; i += istep {
							(*iwork)[i-1] = 1
						}
					}
					Dlaord('D', &mnmin, s, func() *int { y := 1; return &y }())
				}

				for inb = 1; inb <= (*nnb); inb++ {
					//                 Do for each pair of values (NB,NX) in NBVAL and NXVAL.
					nb = (*nbval)[inb-1]
					Xlaenv(1, nb)
					nx = (*nxval)[inb-1]
					Xlaenv(3, nx)

					//                 Save A and its singular values and a copy of
					//                 vector IWORK.
					golapack.Zlacpy('A', &m, &n, copya.CMatrix(lda, opts), &lda, a.CMatrix(lda, opts), &lda)
					Icopy(n, toSlice(iwork, 0), 1, toSlice(iwork, n+1-1), 1)

					//                 Workspace needed.
					lw = nb * (n + 1)

					*srnamt = "ZGEQP3"
					golapack.Zgeqp3(&m, &n, a.CMatrix(lda, opts), &lda, toSlice(iwork, n+1-1), tau, work, &lw, rwork, &info)

					//                 Compute norm(svd(a) - svd(r))
					result.Set(0, Zqrt12(&m, &n, a.CMatrix(lda, opts), &lda, s, work, &lwork, rwork))

					//                 Compute norm( A*P - Q*R )
					result.Set(1, Zqpt01(&m, &n, &mnmin, copya.CMatrix(lda, opts), a.CMatrix(lda, opts), &lda, tau, toSlice(iwork, n+1-1), work, &lwork))

					//                 Compute Q'*Q
					result.Set(2, Zqrt11(&m, &mnmin, a.CMatrix(lda, opts), &lda, tau, work, &lwork))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= ntests; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" %s M =%5d, N =%5d, NB =%4d, _type %2d, test %2d, ratio =%12.5f\n", "ZGEQP3", m, n, nb, imode, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + ntests

				}
			label70:
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
