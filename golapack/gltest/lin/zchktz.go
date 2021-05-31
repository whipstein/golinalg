package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchktz tests ZTZRZF.
func Zchktz(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, thresh *float64, tsterr *bool, a, copya *mat.CVector, s *mat.Vector, tau, work *mat.CVector, rwork *mat.Vector, nout *int, t *testing.T) {
	var eps, one, zero float64
	var i, im, imode, in, info, k, lda, lwork, m, mnmin, mode, n, nerrs, nfail, nrun, ntests, ntypes int

	result := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	ntypes = 3
	ntests = 3
	one = 1.0
	zero = 0.0
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := []byte("ZTZ")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	eps = golapack.Dlamch(Epsilon)

	//     Test the error exits
	if *tsterr {
		Zerrtz(path, t)
	}
	(*infot) = 0

	for im = 1; im <= (*nm); im++ {
		//        Do for each value of M in MVAL.
		m = (*mval)[im-1]
		lda = maxint(1, m)

		for in = 1; in <= (*nn); in++ {
			//           Do for each value of N in NVAL for which M .LE. N.
			n = (*nval)[in-1]
			mnmin = minint(m, n)
			lwork = maxint(1, n*n+4*m+n)

			if m <= n {
				for imode = 1; imode <= ntypes; imode++ {
					if !(*dotype)[imode-1] {
						goto label50
					}

					//                 Do for each _type of singular value distribution.
					//                    0:  zero matrix
					//                    1:  one small singular value
					//                    2:  exponential distribution
					mode = imode - 1

					//                 Test ZTZRQF
					//
					//                 Generate test matrix of size m by n using
					//                 singular value distribution indicated by `mode'.
					if mode == 0 {
						golapack.Zlaset('F', &m, &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), a.CMatrix(lda, opts), &lda)
						for i = 1; i <= mnmin; i++ {
							s.Set(i-1, zero)
						}
					} else {
						matgen.Zlatms(&m, &n, 'U', &iseed, 'N', s, &imode, toPtrf64(one/eps), &one, &m, &n, 'N', a.CMatrix(lda, opts), &lda, work, &info)
						golapack.Zgeqr2(&m, &n, a.CMatrix(lda, opts), &lda, work, work.Off(mnmin+1-1), &info)
						golapack.Zlaset('L', toPtr(m-1), &n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), a.CMatrixOff(1, lda, opts), &lda)
						Dlaord('D', &mnmin, s, func() *int { y := 1; return &y }())
					}

					//                 Save A and its singular values
					golapack.Zlacpy('A', &m, &n, a.CMatrix(lda, opts), &lda, copya.CMatrix(lda, opts), &lda)

					//                 Call ZTZRZF to reduce the upper trapezoidal matrix to
					//                 upper triangular form.
					*srnamt = "ZTZRZF"
					golapack.Ztzrzf(&m, &n, a.CMatrix(lda, opts), &lda, tau, work, &lwork, &info)

					//                 Compute norm(svd(a) - svd(r))
					result.Set(0, Zqrt12(&m, &m, a.CMatrix(lda, opts), &lda, s, work, &lwork, rwork))

					//                 Compute norm( A - R*Q )
					result.Set(1, Zrzt01(&m, &n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts), &lda, tau, work, &lwork))

					//                 Compute norm(Q'*Q - I).
					result.Set(2, Zrzt02(&m, &n, a.CMatrix(lda, opts), &lda, tau, work, &lwork))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= ntests; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							fmt.Printf(" M =%5d, N =%5d, _type %2d, test %2d, ratio =%12.5f\n", m, n, imode, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 3
				label50:
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
