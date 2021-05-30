package lin

import (
	"fmt"
	"golinalg/golapack/gltest"
	"testing"
)

// Zchkqrt tests ZGEQRT and ZGEMQRT.
func Zchkqrt(thresh *float64, tsterr *bool, nm *int, mval *[]int, nn *int, nval *[]int, nnb *int, nbval *[]int, nout *int, _t *testing.T) {
	var i, j, k, m, minmn, n, nb, nerrs, nfail, nrun, ntests, t int

	result := vf(6)
	ntests = 6
	infot := &gltest.Common.Infoc.Infot

	//     Initialize constants
	path := []byte("ZQT")
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if *tsterr {
		Zerrqrt(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M in MVAL.
	for i = 1; i <= (*nm); i++ {
		m = (*mval)[i-1]

		//        Do for each value of N in NVAL.
		for j = 1; j <= (*nn); j++ {
			n = (*nval)[j-1]

			//        Do for each possible value of NB
			minmn = minint(m, n)
			for k = 1; k <= (*nnb); k++ {
				nb = (*nbval)[k-1]

				//              Test ZGEQRT and ZGEMQRT
				if (nb <= minmn) && (nb > 0) {
					Zqrt04(&m, &n, &nb, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
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

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}