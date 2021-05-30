package lin

import (
	"fmt"
	"golinalg/golapack/gltest"
	"testing"
)

// DchkorhrCol tests DORHR_COL using DLATSQR and DGEMQRT. Therefore, DLATSQR
// (used in DGEQR) and DGEMQRT (used in DGEMQR) have to be tested
// before this test.
func DchkorhrCol(thresh *float64, tsterr *bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nout *int, _t *testing.T) {
	var i, imb1, inb1, inb2, j, t, m, n, mb1, nb1, nb2, nfail, nerrs, ntests, nrun int

	result := vf(6)
	infot := &gltest.Common.Infoc.Infot

	ntests = 6

	//     Initialize constants
	path := []byte("DHH")
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if *(tsterr) {
		DerrorhrCol(path, _t)
	}
	*infot = 0

	//     Do for each value of M in mval.
	for i = 1; i <= (*nm); i++ {
		m = (*(mval))[i-(1)]

		//        Do for each value of n in nval.
		for j = 1; j <= (*nn); j++ {
			n = (*(nval))[j-(1)]

			//           Only for M >= n
			if minint(m, n) > 0 && m >= n {
				//              Do for each possible value of mb1
				for imb1 = 1; imb1 <= (*nnb); imb1++ {
					mb1 = (*nbval)[imb1-(1)]

					//                 Only for mb1 > n
					if mb1 > n {
						//                    Do for each possible value of nb1
						for inb1 = 1; inb1 <= (*nnb); inb1++ {
							nb1 = (*nbval)[inb1-(1)]

							//                       Do for each possible value of nb2
							for inb2 = 1; inb2 <= (*nnb); inb2++ {
								nb2 = (*nbval)[inb2-(1)]

								if nb1 > 0 && nb2 > 0 {
									//                             Test DORHR_COL
									DorhrCol01(&m, &n, &mb1, &nb1, &nb2, result)

									//                             Print information about the tests that did
									//                             not pass the threshold.
									for t = 1; t <= ntests; t++ {
										if result.Get(t-1) >= (*thresh) {
											if nfail == 0 && nerrs == 0 {
												Alahd(path)
											}
											fmt.Printf("M=%5d, n=%5d, mb1=%5d, nb1=%5d, nb2=%5d test(%2d)=%12.5f\n", m, n, mb1, nb1, nb2, t, result.Get(t-1))
											nfail = nfail + 1
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

	//     Verify number of tests match original.
	tgtRuns := 7950
	if nrun != tgtRuns {
		_t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
