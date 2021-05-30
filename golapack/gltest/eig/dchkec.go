package eig

import (
	"fmt"
	"golinalg/golapack"
	"testing"
)

// Dchkec tests eigen- condition estimation routines
//        DLALN2, DLASY2, DLANV2, DLAQTR, DLAEXC,
//        DTRSYL, DTREXC, DTRSNA, DTRSEN
//
// In all cases, the routine runs through a fixed set of numerical
// examples, subjects them to various tests, and compares the test
// results to a threshold THRESH. In addition, DTREXC, DTRSNA and DTRSEN
// are tested by reading in precomputed examples from a file (on input
// unit NIN).  Output is written to output unit NOUT.
func Dchkec(thresh *float64, tsterr *bool, t *testing.T) {
	var ok bool
	var eps, rlaexc, rlaln2, rlanv2, rlaqtr, rlasy2, rtrexc, rtrsyl, sfmin float64
	var klaexc, klaln2, klanv2, klaqtr, klasy2, ktrexc, ktrsen, ktrsna, ktrsyl, llaexc, llaln2, llanv2, llaqtr, llasy2, ltrexc, ltrsyl, nlanv2, nlaqtr, nlasy2, ntests, ntrsyl int

	rtrsen := vf(3)
	rtrsna := vf(3)
	ltrsen := make([]int, 3)
	ltrsna := make([]int, 3)
	nlaexc := make([]int, 2)
	nlaln2 := make([]int, 2)
	ntrexc := make([]int, 3)
	ntrsen := make([]int, 3)
	ntrsna := make([]int, 3)

	path := []byte("DEC")
	eps = golapack.Dlamch(Precision)
	sfmin = golapack.Dlamch(SafeMinimum)

	//     Print header information
	fmt.Printf(" Tests of the Nonsymmetric eigenproblem condition estimation routines\n DLALN2, DLASY2, DLANV2, DLAEXC, DTRSYL, DTREXC, DTRSNA, DTRSEN, DLAQTR\n\n")
	fmt.Printf(" Relative machine precision (EPS) = %16.6E\n Safe minimum (SFMIN)             = %16.6E\n\n", eps, sfmin)
	fmt.Printf(" Routines pass computational tests if test ratio is less than%8.2f\n\n\n", *thresh)

	//     Test error exits if TSTERR is .TRUE.
	if *tsterr {
		Derrec(path, t)
	}

	ok = true
	Dget31(&rlaln2, &llaln2, &nlaln2, &klaln2)
	if rlaln2 > (*thresh) || nlaln2[0] != 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DLALN2: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rlaln2, llaln2, nlaln2, klaln2)
	}

	Dget32(&rlasy2, &llasy2, &nlasy2, &klasy2)
	if rlasy2 > (*thresh) {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DLASY2: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rlasy2, llasy2, nlasy2, klasy2)
	}

	Dget33(&rlanv2, &llanv2, &nlanv2, &klanv2)
	if rlanv2 > (*thresh) || nlanv2 != 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DLANV2: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rlanv2, llanv2, nlanv2, klanv2)
	}

	Dget34(&rlaexc, &llaexc, &nlaexc, &klaexc)
	if rlaexc > (*thresh) || nlaexc[1] != 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DLAEXC: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rlaexc, llaexc, nlaexc, klaexc)
	}

	Dget35(&rtrsyl, &ltrsyl, &ntrsyl, &ktrsyl)
	if rtrsyl > (*thresh) {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DTRSYL: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsyl, ltrsyl, ntrsyl, ktrsyl)
	}

	Dget36(&rtrexc, &ltrexc, &ntrexc, &ktrexc)
	if rtrexc > (*thresh) || ntrexc[2] > 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DTREXC: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrexc, ltrexc, ntrexc, ktrexc)
	}

	Dget37(rtrsna, &ltrsna, &ntrsna, &ktrsna)
	if rtrsna.Get(0) > (*thresh) || rtrsna.Get(1) > (*thresh) || ntrsna[0] != 0 || ntrsna[1] != 0 || ntrsna[2] != 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DTRSNA: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsna, ltrsna, ntrsna, ktrsna)
	}

	Dget38(rtrsen, &ltrsen, &ntrsen, &ktrsen)
	if rtrsen.Get(0) > (*thresh) || rtrsen.Get(1) > (*thresh) || ntrsen[0] != 0 || ntrsen[1] != 0 || ntrsen[2] != 0 {
		ok = false
		t.Fail()
		fmt.Printf(" Error in DTRSEN: RMAX =%v\n LMAX = %8d NINFO=%8d KNT=%8d\n", rtrsen, ltrsen, ntrsen, ktrsen)
	}

	Dget39(&rlaqtr, &llaqtr, &nlaqtr, &klaqtr)
	if rlaqtr > (*thresh) {
		t.Fail()
		ok = false
		fmt.Printf(" Error in DLAQTR: RMAX =%12.3E\n LMAX = %8d NINFO=%8d KNT=%8d\n", rlaqtr, llaqtr, nlaqtr, klaqtr)
	}

	ntests = klaln2 + klasy2 + klanv2 + klaexc + ktrsyl + ktrexc + ktrsna + ktrsen + klaqtr
	if ok {
		fmt.Printf("\n All tests for %3s routines passed the threshold ( %6d tests run)\n", path, ntests)
	}
}
