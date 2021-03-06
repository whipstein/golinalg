package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// dchklqt tests DGELQT and DGEMLQT.
func dchklqt(thresh float64, tsterr bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, _t *testing.T) {
	var i, j, k, m, minmn, n, nb, nerrs, nfail, nrun, ntests, t int

	result := vf(6)
	infot := &gltest.Common.Infoc.Infot

	ntests = 6

	//     Initialize constants
	path := "Dtq"
	alasumStart(path)
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if tsterr {
		derrlqt(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M in MVAL.
	for i = 1; i <= nm; i++ {
		m = mval[i-1]

		//        Do for each value of N in NVAL.
		for j = 1; j <= nn; j++ {
			n = nval[j-1]

			//        Do for each possible value of NB
			minmn = min(m, n)
			for k = 1; k <= nnb; k++ {
				nb = nbval[k-1]

				//              Test DGELQT and DGEMLQT
				if (nb <= minmn) && (nb > 0) {
					dlqt04(m, n, nb, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for t = 1; t <= ntests; t++ {
						if result.Get(t-1) >= thresh {
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							_t.Fail()
							fmt.Printf(" M=%5d, N=%5d, NB=%4d test(%2d)=%12.5f\n", m, n, nb, t, result.Get(t-1))
							nfail++
						}
					}
					nrun += ntests
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 510
	if nrun != tgtRuns {
		_t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, nerrs)
	alasumEnd(nfail, nrun, nerrs)
}
