package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchktz tests Ztzrzf.
func zchktz(dotype []bool, nm int, mval []int, nn int, nval []int, thresh *float64, tsterr *bool, a, copya *mat.CVector, s *mat.Vector, tau, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var eps, one, zero float64
	var i, im, imode, in, k, lda, lwork, m, mnmin, mode, n, nerrs, nfail, nrun, ntests, ntypes int
	var err error

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
	path := "Ztz"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	eps = golapack.Dlamch(Epsilon)

	//     Test the error exits
	if *tsterr {
		zerrtz(path, t)
	}
	(*infot) = 0

	for im = 1; im <= nm; im++ {
		//        Do for each value of M in MVAL.
		m = mval[im-1]
		lda = max(1, m)

		for in = 1; in <= nn; in++ {
			//           Do for each value of N in NVAL for which M .LE. N.
			n = nval[in-1]
			mnmin = min(m, n)
			lwork = max(1, n*n+4*m+n)

			if m <= n {
				for imode = 1; imode <= ntypes; imode++ {
					if !dotype[imode-1] {
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
						golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), a.CMatrix(lda, opts))
						for i = 1; i <= mnmin; i++ {
							s.Set(i-1, zero)
						}
					} else {
						if err = matgen.Zlatms(m, n, 'U', &iseed, 'N', s, imode, one/eps, one, m, n, 'N', a.CMatrix(lda, opts), work); err != nil {
							panic(err)
						}
						if err = golapack.Zgeqr2(m, n, a.CMatrix(lda, opts), work, work.Off(mnmin)); err != nil {
							panic(err)
						}
						golapack.Zlaset(Lower, m-1, n, complex(zero, 0), complex(zero, 0), a.Off(1).CMatrix(lda, opts))
						dlaord('D', mnmin, s, 1)
					}

					//                 Save A and its singular values
					golapack.Zlacpy(Full, m, n, a.CMatrix(lda, opts), copya.CMatrix(lda, opts))

					//                 Call Ztzrzf to reduce the upper trapezoidal matrix to
					//                 upper triangular form.
					*srnamt = "Ztzrzf"
					if err = golapack.Ztzrzf(m, n, a.CMatrix(lda, opts), tau, work, lwork); err != nil {
						panic(err)
					}

					//                 Compute norm(svd(a) - svd(r))
					result.Set(0, zqrt12(m, m, a.CMatrix(lda, opts), s, work, lwork, rwork))

					//                 Compute norm( A - R*Q )
					result.Set(1, zrzt01(m, n, copya.CMatrix(lda, opts), a.CMatrix(lda, opts), tau, work, lwork))

					//                 Compute norm(Q'*Q - I).
					result.Set(2, zrzt02(m, n, a.CMatrix(lda, opts), tau, work, lwork))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= ntests; k++ {
						if result.Get(k-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" m=%5d, n=%5d, _type %2d, test %2d, ratio =%12.5f\n", m, n, imode, k, result.Get(k-1))
							nfail++
						}
					}
					nrun += 3
				label50:
				}
			}
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
