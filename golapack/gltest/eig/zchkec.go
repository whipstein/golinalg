package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
)

// zchkec tests eigen- condition estimation routines
//        Ztrsyl, CTREXC, CTRSNA, CTRSEN
//
// In all cases, the routine runs through a fixed set of numerical
// examples, subjects them to various tests, and compares the test
// results to a threshold THRESH. In addition, Ztrsna and CTRSEN are
// tested by reading in precomputed examples from a file (on input unit
// NIN).  Output is written to output unit NOUT.
func zchkec(thresh float64, tsterr bool, t *testing.T) (ntests int) {
	// var ok bool
	var eps, rtrexc, rtrsyl, sfmin float64
	_ = eps
	_ = sfmin
	var ktrexc, ktrsen, ktrsna, ktrsyl, ltrexc, ltrsyl, ntrexc, ntrsyl int
	rtrsen := vf(3)
	rtrsna := vf(3)
	ltrsen := make([]int, 3)
	ltrsna := make([]int, 3)
	ntrsen := make([]int, 3)
	ntrsna := make([]int, 3)

	// path := "Zec"
	eps = golapack.Dlamch(Precision)
	sfmin = golapack.Dlamch(SafeMinimum)
	// fmt.Printf(" Tests of the Nonsymmetric eigenproblem condition estimation routines\n Ztrsyl, Ztrexc, Ztrsna, Ztrsen\n\n")
	// fmt.Printf(" Relative machine precision (eps) = %16.6E\n Safe minimum (sfmin)             = %16.6E\n\n", eps, sfmin)
	// fmt.Printf(" Routines pass computational tests if test ratio is less than%8.2f\n\n\n", thresh)

	//     Test error exits if TSTERR is .TRUE.
	// if tsterr {
	// 	zerrec(path, t)
	// }

	// ok = true
	rtrsyl, ltrsyl, ntrsyl, ktrsyl = zget35()
	if rtrsyl > thresh {
		t.Fail()
		// ok = false
		fmt.Printf(" Error in Ztrsyl: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsyl, ltrsyl, ntrsyl, ktrsyl)
	}

	rtrexc, ltrexc, ntrexc, ktrexc = zget36()
	if rtrexc > thresh || ntrexc > 0 {
		t.Fail()
		// ok = false
		fmt.Printf(" Error in Ztrexc: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrexc, ltrexc, ntrexc, ktrexc)
	}

	ktrsna = zget37(rtrsna, &ltrsna, &ntrsna)
	if rtrsna.Get(0) > thresh || rtrsna.Get(1) > thresh || ntrsna[0] != 0 || ntrsna[1] != 0 || ntrsna[2] != 0 {
		t.Fail()
		// ok = false
		fmt.Printf(" Error in Ztrsna: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsna, ltrsna, ntrsna, ktrsna)
	}

	ktrsen = zget38(rtrsen, &ltrsen, &ntrsen)
	if rtrsen.Get(0) > thresh || rtrsen.Get(1) > thresh || ntrsen[0] != 0 || ntrsen[1] != 0 || ntrsen[2] != 0 {
		t.Fail()
		// ok = false
		fmt.Printf(" Error in Ztrsen: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsen, ltrsen, ntrsen, ktrsen)
	}

	ntests = ktrsyl + ktrexc + ktrsna + ktrsen
	// if ok {
	// 	fmt.Printf("\n All tests for %3s routines passed the threshold ( %6d tests run)\n", path, ntests)
	// }

	return
}
