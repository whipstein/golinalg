package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
)

// Dchkec tests eigen- condition estimation routines
//        Dlaln2, Dlasy2, Dlanv2, Dlaqtr, Dlaexc,
//        Dtrsyl, Dtrexc, Dtrsna, Dtrsen
//
// In all cases, the routine runs through a fixed set of numerical
// examples, subjects them to various tests, and compares the test
// results to a threshold THRESH. In addition, Dtrexc, Dtrsna and Dtrsen
// are tested by reading in precomputed examples from a file (on input
// unit NIN).  Output is written to output unit NOUT.
func Dchkec(thresh *float64, tsterr *bool, t *testing.T) (ntests int) {
	// var ok bool
	var eps, rlaexc, rlaln2, rlanv2, rlaqtr, rlasy2, rtrexc, rtrsyl, sfmin float64
	_ = eps
	_ = sfmin
	var klaexc, klaln2, klanv2, klaqtr, klasy2, ktrexc, ktrsen, ktrsna, ktrsyl, llaexc, llaln2, llanv2, llaqtr, llasy2, ltrexc, ltrsyl, nlanv2, nlaqtr, nlasy2, ntrsyl int

	rtrsen := vf(3)
	rtrsna := vf(3)
	ltrsen := make([]int, 3)
	ltrsna := make([]int, 3)
	nlaexc := make([]int, 2)
	nlaln2 := make([]int, 2)
	ntrexc := make([]int, 3)
	ntrsen := make([]int, 3)
	ntrsna := make([]int, 3)

	// path := "Dec"
	eps = golapack.Dlamch(Precision)
	sfmin = golapack.Dlamch(SafeMinimum)

	//     Print header information
	// fmt.Printf(" Tests of the Nonsymmetric eigenproblem condition estimation routines\n Dlaln2, Dlasy2, Dlanv2, Dlaexc, Dtrsyl, Dtrexc, Dtrsna, Dtrsen, Dlaqtr\n\n")
	// fmt.Printf(" Relative machine precision (EPS) = %16.6E\n Safe minimum (SFMIN)             = %16.6E\n\n", eps, sfmin)
	// fmt.Printf(" Routines pass computational tests if test ratio is less than%8.2f\n\n\n", *thresh)

	//     Test error exits if TSTERR is .TRUE.
	// if *tsterr {
	// 	derrec(path, t)
	// }

	// ok = true
	rlaln2, llaln2, klaln2 = dget31(&nlaln2)
	if rlaln2 > (*thresh) || nlaln2[0] != 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dlaln2: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rlaln2, llaln2, nlaln2, klaln2)
	}

	rlasy2, llasy2, nlasy2, klasy2 = dget32()
	if rlasy2 > (*thresh) {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dlasy2: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rlasy2, llasy2, nlasy2, klasy2)
	}

	rlanv2, llanv2, nlanv2, klanv2 = dget33()
	if rlanv2 > (*thresh) || nlanv2 != 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dlanv2: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rlanv2, llanv2, nlanv2, klanv2)
	}

	rlaexc, llaexc, klaexc = dget34(&nlaexc)
	if rlaexc > (*thresh) || nlaexc[1] != 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dlaexc: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rlaexc, llaexc, nlaexc, klaexc)
	}

	rtrsyl, ltrsyl, ntrsyl, ktrsyl = dget35()
	if rtrsyl > (*thresh) {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dtrsyl: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rtrsyl, ltrsyl, ntrsyl, ktrsyl)
	}

	rtrexc, ltrexc, ktrexc = dget36(&ntrexc)
	if rtrexc > (*thresh) || ntrexc[2] > 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dtrexc: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rtrexc, ltrexc, ntrexc, ktrexc)
	}

	ktrsna = dget37(rtrsna, &ltrsna, &ntrsna)
	if rtrsna.Get(0) > (*thresh) || rtrsna.Get(1) > (*thresh) || ntrsna[0] != 0 || ntrsna[1] != 0 || ntrsna[2] != 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dtrsna: rmax=%v\n lmax=%8d ninfo=%8d knt=%8d\n", rtrsna, ltrsna, ntrsna, ktrsna)
	}

	ktrsen = dget38(rtrsen, &ltrsen, &ntrsen)
	if rtrsen.Get(0) > (*thresh) || rtrsen.Get(1) > (*thresh) || ntrsen[0] != 0 || ntrsen[1] != 0 || ntrsen[2] != 0 {
		// ok = false
		t.Fail()
		fmt.Printf(" Error in Dtrsen: rmax=%v\n lmax=%8d ninfo=%8d knt=%8d\n", rtrsen.Data(), ltrsen, ntrsen, ktrsen)
	}

	rlaqtr, llaqtr, nlaqtr, klaqtr = dget39()
	if rlaqtr > (*thresh) {
		t.Fail()
		// ok = false
		fmt.Printf(" Error in Dlaqtr: rmax=%12.3E\n lmax=%8d ninfo=%8d knt=%8d\n", rlaqtr, llaqtr, nlaqtr, klaqtr)
	}

	ntests = klaln2 + klasy2 + klanv2 + klaexc + ktrsyl + ktrexc + ktrsna + ktrsen + klaqtr
	// if ok {
	// 	fmt.Printf("\n All tests for %3s routines passed the threshold ( %6d tests run)\n", path, ntests)
	// }

	return
}
