package eig

import "fmt"

// alasvm prints a summary of results from one of the -DRV- routines.
func alasvm(_type string, nfail, nrun, nerrs int) {
	if nfail > 0 {
		fmt.Printf(" %3s drivers: %6d out of %6d tests failed to pass the threshold\n", _type, nfail, nrun)
	} else {
		fmt.Printf(" All tests for %3s drivers  passed the threshold ( %6d tests run)\n", _type, nrun)
	}
	if nerrs > 0 {
		fmt.Printf("              %6d error messages recorded\n", nerrs)
	}
}
