package eig

import "fmt"

// Alasum prints a summary of results from one of the -CHK- routines.
func Alasum(_type []byte, nfail, nrun, nerrs *int) {
	if *nfail > 0 {
		fmt.Printf(" %3s: %6d out of %6d tests failed to pass the threshold\n", _type, *nfail, *nrun)
	} else {
		fmt.Printf(" All tests for %3s routines passed the threshold ( %6d tests run)\n", _type, *nrun)
	}
	if *nerrs > 0 {
		fmt.Printf("      %6d error messages recorded\n", *nerrs)
	}
}
