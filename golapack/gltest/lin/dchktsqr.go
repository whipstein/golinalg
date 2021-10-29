package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// dchkqrt tests DGETSQR and DORMTSQR.
func dchktsqr(thresh float64, tsterr bool, nm int, mval []int, nn int, nval []int, nnb int, nbval []int, _t *testing.T) {
	var i, imb, inb, j, m, mb, n, nb, nerrs, nfail, nrun, ntests, t int

	result := vf(6)
	infot := &gltest.Common.Infoc.Infot

	ntests = 6

	//     Initialize constants
	path := "Dts"
	nrun = 0
	nfail = 0
	nerrs = 0

	//     Test the error exits
	if tsterr {
		derrtsqr(path, _t)
	}
	(*infot) = 0

	//     Do for each value of M in MVAL.
	for i = 1; i <= nm; i++ {
		m = mval[i-1]

		//        Do for each value of N in NVAL.
		for j = 1; j <= nn; j++ {
			n = nval[j-1]
			if min(m, n) != 0 {
				for inb = 1; inb <= nnb; inb++ {
					mb = nbval[inb-1]
					xlaenv(1, mb)
					for imb = 1; imb <= nnb; imb++ {
						nb = nbval[imb-1]
						xlaenv(2, nb)

						//                 Test DGEQR and DGEMQR
						dtsqr01('T', m, n, mb, nb, result)

						//                 Print information about the tests that did not
						//                 pass the threshold.
						for t = 1; t <= ntests; t++ {
							if result.Get(t-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								_t.Fail()
								fmt.Printf("TS: M=%5d, N=%5d, MB=%5d, NB=%5d test(%2d)=%12.5f\n", m, n, mb, nb, t, result.Get(t-1))
								nfail++
							}
						}
						nrun += ntests
					}
				}
			}
		}
	}

	//     Do for each value of M in MVAL.
	for i = 1; i <= nm; i++ {
		m = mval[i-1]

		//        Do for each value of N in NVAL.
		for j = 1; j <= nn; j++ {
			n = nval[j-1]
			if min(m, n) != 0 {
				for inb = 1; inb <= nnb; inb++ {
					mb = nbval[inb-1]
					xlaenv(1, mb)
					for imb = 1; imb <= nnb; imb++ {
						nb = nbval[imb-1]
						xlaenv(2, nb)

						//                 Test DGEQR and DGEMQR
						dtsqr01('S', m, n, mb, nb, result)

						//                 Print information about the tests that did not
						//                 pass the threshold.
						for t = 1; t <= ntests; t++ {
							if result.Get(t-1) >= thresh {
								if nfail == 0 && nerrs == 0 {
									alahd(path)
								}
								_t.Fail()
								fmt.Printf("SW: M=%5d, N=%5d, MB=%5d, NB=%5d test(%2d)=%12.5f\n", m, n, mb, nb, t, result.Get(t-1))
								nfail++
							}
						}
						nrun += ntests
					}
				}
			}
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 10800
	if nrun != tgtRuns {
		_t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
