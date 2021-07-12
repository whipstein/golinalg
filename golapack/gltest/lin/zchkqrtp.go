package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zchkqrtp tests ZTPQRT and ZTPMQRT.
func Zchkqrtp(thresh *float64, tsterr *bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nout *int, _t *testing.T) {
	var i, j, k, l, m, minmn, n, nb, nerrs, nfail, nrun, ntests, t int

	result := vf(6)
	ntests = 6
	infot := &gltest.Common.Infoc.Infot

	//     Initialize constants
	path := []byte("ZQX")
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if *tsterr {
		Zerrqrtp(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M
	for i = 1; i <= (*nm); i++ {
		m = (*mval)[i-1]

		//        Do for each value of N
		for j = 1; j <= (*nn); j++ {
			n = (*nval)[j-1]

			//           Do for each value of L
			minmn = min(m, n)
			for l = 0; l <= minmn; l += max(minmn, 1) {
				//              Do for each possible value of NB
				for k = 1; k <= (*nnb); k++ {
					nb = (*nbval)[k-1]

					//                 Test ZTPQRT and ZTPMQRT
					if (nb <= n) && (nb > 0) {
						Zqrt05(&m, &n, &l, &nb, result)

						//                    Print information about the tests that did not
						//                    pass the threshold.
						for t = 1; t <= ntests; t++ {
							if result.Get(t-1) >= (*thresh) {
								_t.Fail()
								if nfail == 0 && nerrs == 0 {
									Alahd(path)
								}
								fmt.Printf(" M=%5d, N=%5d, NB=%4d test(%2d)=%12.5f\n", m, n, nb, t, result.Get(t-1))
								nfail = nfail + 1
							}
						}
						nrun = nrun + ntests
					}
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
