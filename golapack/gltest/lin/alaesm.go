package lin

import "fmt"

// alaesm prints a summary of results from one of the -ERR- routines.
func alaesm(path string, ok bool) {
	if ok {
		fmt.Printf(" %3s routines passed the tests of the error exits\n", path)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
