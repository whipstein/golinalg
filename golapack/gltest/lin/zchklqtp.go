package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// zchklqtp tests ZTPLQT and ZTPMLQT.
func zchklqtp(thresh float64, tsterr bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, _t *testing.T) {
	var i, j, k, l, m, minmn, n, nb, nerrs, nfail, nrun, ntests, t int

	result := vf(6)

	ntests = 6
	infot := &gltest.Common.Infoc.Infot

	//     Initialize constants
	path := "Zxq"
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if tsterr {
		zerrlqtp(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M
	for i = 1; i <= nm; i++ {
		m = mval[i-1]

		//        Do for each value of N
		for j = 1; j <= nn; j++ {
			n = nval[j-1]

			//           Do for each value of L
			minmn = min(m, n)
			for l = 0; l <= minmn; l += max(minmn, 1) {
				//              Do for each possible value of NB
				for k = 1; k <= nnb; k++ {
					nb = nbval[k-1]

					//                 Test DTPLQT and DTPMLQT
					if (nb <= m) && (nb > 0) {
						zlqt05(m, n, l, nb, result)

						//                    Print information about the tests that did not
						//                    pass the threshold.
						for t = 1; t <= ntests; t++ {
							if result.Get(t-1) >= thresh {
								_t.Fail()
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								fmt.Printf(" M=%5d, N=%5d, NB=%4d L=%4d test(%2d)=%12.5f\n", m, n, nb, l, t, result.Get(t-1))
								nfail++
							}
						}
						nrun = nrun + ntests
					}
				}
			}
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
