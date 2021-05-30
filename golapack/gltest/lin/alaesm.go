package lin

import "fmt"

// Alaesm prints a summary of results from one of the -ERR- routines.
func Alaesm(path []byte, ok *bool) {
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits\n", path)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
