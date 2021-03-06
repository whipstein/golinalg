package eig

import "fmt"

// dlasum prints a summary of the results from one of the test routines.
func dlasum(_type string, ie, nrun int) {
	if ie > 0 {
		fmt.Printf(" %3s%2s%4d%8s%5d%35s\n", _type, ": ", ie, " out of ", nrun, " tests failed to pass the threshold")
	} else {
		// fmt.Printf(" %3s%s%5d%s\n", _type, " passed ( ", nrun, " tests run)")
		fmt.Printf(" routines:    Passed (%5d tests run)\n", nrun)
	}
}

func dlasumStart(_type string) {
	fmt.Printf(" Testing %3s routines.........", _type)
}

func dlasumEnd(nfail, nrun int) {
	if nfail > 0 {
		fmt.Printf("Fail\t\t( %6d tests failed )\n", nfail)
	} else {
		fmt.Printf("Pass\t\t( %6d tests run )\n", nrun)
	}
}
