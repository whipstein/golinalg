package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"testing"
)

// Dchktz tests DTZRZF.
func Dchktz(dotype *[]bool, nm *int, mval *[]int, nn *int, nval *[]int, thresh *float64, tsterr *bool, a, copya, s, tau, work *mat.Vector, nout *int, t *testing.T) {
	var eps, one, zero float64
	var i, im, imode, in, info, k, lda, lwork, m, mnmin, mode, n, nerrs, nfail, nrun, ntests, ntypes int

	result := vf(3)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	ntypes = 3
	ntests = 3
	one = 1.0
	zero = 0.0

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := []byte("DTZ")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}
	eps = golapack.Dlamch(Epsilon)

	//     Test the error exits
	if *tsterr {
		Derrtz(path, t)
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
			lwork = maxint(1, n*n+4*m+n, m*n+2*mnmin+4*n)

			if m <= n {
				for imode = 1; imode <= ntypes; imode++ {
					if !(*dotype)[imode-1] {
						goto label50
					}

					//                 Do for each type of singular value distribution.
					//                    0:  zero matrix
					//                    1:  one small singular value
					//                    2:  exponential distribution
					mode = imode - 1

					//                 Test DTZRQF
					//
					//                 Generate test matrix of size m by n using
					//                 singular value distribution indicated by `mode'.
					if mode == 0 {
						golapack.Dlaset('F', &m, &n, &zero, &zero, a.Matrix(lda, opts), &lda)
						for i = 1; i <= mnmin; i++ {
							s.Set(i-1, zero)
						}
					} else {
						matgen.Dlatms(&m, &n, 'U', &iseed, 'N', s, &imode, func() *float64 { y := one / eps; return &y }(), &one, &m, &n, 'N', a.Matrix(lda, opts), &lda, work, &info)
						golapack.Dgeqr2(&m, &n, a.Matrix(lda, opts), &lda, work, work.Off(mnmin+1-1), &info)
						golapack.Dlaset('L', toPtr(m-1), &n, &zero, &zero, a.MatrixOff(1, lda, opts), &lda)
						Dlaord('D', &mnmin, s, func() *int { y := 1; return &y }())
					}

					//                 Save A and its singular values
					golapack.Dlacpy('F', &m, &n, a.Matrix(lda, opts), &lda, copya.Matrix(lda, opts), &lda)

					//                 Call DTZRZF to reduce the upper trapezoidal matrix to
					//                 upper triangular form.
					*srnamt = "DTZRZF"
					golapack.Dtzrzf(&m, &n, a.Matrix(lda, opts), &lda, tau, work, &lwork, &info)

					//                 Compute norm(svd(a) - svd(r))
					result.Set(0, Dqrt12(&m, &m, a.Matrix(lda, opts), &lda, s, work, &lwork))

					//                 Compute norm( A - R*Q )
					result.Set(1, Drzt01(&m, &n, copya.Matrix(lda, opts), a.Matrix(lda, opts), &lda, tau, work, &lwork))

					//                 Compute norm(Q'*Q - I).
					result.Set(2, Drzt02(&m, &n, a.Matrix(lda, opts), &lda, tau, work, &lwork))

					//                 Print information about the tests that did not pass
					//                 the threshold.
					for k = 1; k <= ntests; k++ {
						if result.Get(k-1) >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
							fmt.Printf(" M =%5d, N =%5d, type %2d, test %2d, ratio =%12.5f\n", m, n, imode, k, result.Get(k-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + 3
				label50:
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 252
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
