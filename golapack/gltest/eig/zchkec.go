package eig

import (
	"fmt"
	"golinalg/golapack"
	"testing"
)

// Zchkec tests eigen- condition estimation routines
//        ZTRSYL, CTREXC, CTRSNA, CTRSEN
//
// In all cases, the routine runs through a fixed set of numerical
// examples, subjects them to various tests, and compares the test
// results to a threshold THRESH. In addition, ZTRSNA and CTRSEN are
// tested by reading in precomputed examples from a file (on input unit
// NIN).  Output is written to output unit NOUT.
func Zchkec(thresh *float64, tsterr *bool, t *testing.T) {
	var ok bool
	var eps, rtrexc, rtrsyl, sfmin float64
	var ktrexc, ktrsen, ktrsna, ktrsyl, ltrexc, ltrsyl, ntests, ntrexc, ntrsyl int
	rtrsen := vf(3)
	rtrsna := vf(3)
	ltrsen := make([]int, 3)
	ltrsna := make([]int, 3)
	ntrsen := make([]int, 3)
	ntrsna := make([]int, 3)

	path := []byte("ZEC")
	eps = golapack.Dlamch(Precision)
	sfmin = golapack.Dlamch(SafeMinimum)
	fmt.Printf(" Tests of the Nonsymmetric eigenproblem condition estimation routines\n ZTRSYL, ZTREXC, ZTRSNA, ZTRSEN\n\n")
	fmt.Printf(" Relative machine precision (EPS) = %16.6E\n Safe minimum (SFMIN)             = %16.6E\n\n", eps, sfmin)
	fmt.Printf(" Routines pass computational tests if test ratio is less than%8.2f\n\n\n", *thresh)

	//     Test error exits if TSTERR is .TRUE.
	if *tsterr {
		Zerrec(path, t)
	}

	ok = true
	Zget35(&rtrsyl, &ltrsyl, &ntrsyl, &ktrsyl, t)
	if rtrsyl > (*thresh) {
		t.Fail()
		ok = false
		fmt.Printf(" Error in ZTRSYL: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsyl, ltrsyl, ntrsyl, ktrsyl)
	}

	Zget36(&rtrexc, &ltrexc, &ntrexc, &ktrexc, t)
	if rtrexc > (*thresh) || ntrexc > 0 {
		t.Fail()
		ok = false
		fmt.Printf(" Error in ZTREXC: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrexc, ltrexc, ntrexc, ktrexc)
	}

	Zget37(rtrsna, &ltrsna, &ntrsna, &ktrsna, t)
	if rtrsna.Get(0) > (*thresh) || rtrsna.Get(1) > (*thresh) || ntrsna[0] != 0 || ntrsna[1] != 0 || ntrsna[2] != 0 {
		t.Fail()
		ok = false
		fmt.Printf(" Error in ZTRSNA: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsna, ltrsna, ntrsna, ktrsna)
	}

	Zget38(rtrsen, &ltrsen, &ntrsen, &ktrsen, t)
	if rtrsen.Get(0) > (*thresh) || rtrsen.Get(1) > (*thresh) || ntrsen[0] != 0 || ntrsen[1] != 0 || ntrsen[2] != 0 {
		t.Fail()
		ok = false
		fmt.Printf(" Error in ZTRSEN: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsen, ltrsen, ntrsen, ktrsen)
	}

	ntests = ktrsyl + ktrexc + ktrsna + ktrsen
	if ok {
		fmt.Printf("\n All tests for %3s routines passed the threshold ( %6d tests run)\n", path, ntests)
	}
}
