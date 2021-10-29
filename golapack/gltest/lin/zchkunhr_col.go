package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// zchkunhrCol tests ZUNHR_COL using ZLATSQR and ZGEMQRT. Therefore, ZLATSQR
// (used in ZGEQR) and ZGEMQRT (used in ZGEMQR) have to be tested
// before this test.
func zchkunhrCol(thresh float64, tsterr bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, _t *testing.T) {
	var i, imb1, inb1, inb2, j, m, mb1, n, nb1, nb2, nerrs, nfail, nrun, ntests, t int

	result := vf(6)

	ntests = 6
	infot := &gltest.Common.Infoc.Infot

	//     Initialize constants
	path := "Zhh"
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if tsterr {
		zerrunhrCol(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M in MVAL.
	for i = 1; i <= nm; i++ {
		m = mval[i-1]

		//        Do for each value of N in NVAL.
		for j = 1; j <= nn; j++ {
			n = nval[j-1]

			//           Only for M >= N
			if min(m, n) > 0 && m >= n {
				//              Do for each possible value of mb1
				for imb1 = 1; imb1 <= nnb; imb1++ {
					mb1 = nbval[imb1-1]

					//                 Only for mb1 > N
					if mb1 > n {
						//                    Do for each possible value of nb1
						for inb1 = 1; inb1 <= nnb; inb1++ {
							nb1 = nbval[inb1-1]

							//                       Do for each possible value of nb2
							for inb2 = 1; inb2 <= nnb; inb2++ {
								nb2 = nbval[inb2-1]

								if nb1 > 0 && nb2 > 0 {
									//                             Test ZUNHR_COL
									zunhrCol01(m, n, mb1, nb1, nb2, result)

									//                             Print information about the tests that did
									//                             not pass the threshold.
									for t = 1; t <= ntests; t++ {
										if result.Get(t-1) >= thresh {
											_t.Fail()
											if nfail == 0 && nerrs == 0 {
												alahd(path)
											}
											fmt.Printf("m=%5d, n=%5d, mb1=%5d, nb1=%5d, nb2=%5d test(%2d)=%12.5f\n", m, n, mb1, nb1, nb2, t, result.Get(t-1))
											nfail++
										}
									}
									nrun = nrun + ntests
								}
							}
						}
					}
				}
			}
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
