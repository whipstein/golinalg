package eig

import "fmt"

// alasum prints a summary of results from one of the -CHK- routines.
func alasum(_type string, nfail, nrun, nerrs int) {
	if nfail > 0 {
		fmt.Printf(" %3s: %6d out of %6d tests failed to pass the threshold\n", _type, nfail, nrun)
	} else {
		// fmt.Printf(" All tests for %3s routines passed the threshold ( %6d tests run)\n", _type, nrun)
		fmt.Printf(" routines Passed ( %6d tests run)\n", nrun)
	}
	if nerrs > 0 {
		fmt.Printf("      %6d error messages recorded\n", nerrs)
	}
}
