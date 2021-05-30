package eig

import "fmt"

// Dlasum prints a summary of the results from one of the test routines.
func Dlasum(_type []byte, ie *int, nrun *int) {
	if (*ie) > 0 {
		fmt.Printf(" %3s%2s%4d%8s%5d%35s\n", _type, ": ", *ie, " out of ", *nrun, " tests failed to pass the threshold")
	} else {
		fmt.Printf(" %14s%3s%24s%5d%11s\n", "All tests for ", _type, " passed the threshold ( ", *nrun, " tests run)")
	}
}
