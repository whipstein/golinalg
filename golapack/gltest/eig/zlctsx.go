package eig

import "github.com/whipstein/golinalg/golapack/gltest"

// zlctsx function is used to determine what eigenvalues will be
// selected.  If this is part of the test driver ZDRGSX, do not
// change the code UNLESS you are testing input examples and not
// using the built-in examples.
func zlctsx(alpha, beta complex128) (zlctsxReturn bool) {
	m := &gltest.Common.Mn.M
	n := &gltest.Common.Mn.N
	mplusn := &gltest.Common.Mn.Mplusn
	i := &gltest.Common.Mn.I
	fs := &gltest.Common.Mn.Fs

	if *fs {
		(*i) = (*i) + 1
		if (*i) <= (*m) {
			zlctsxReturn = false
		} else {
			zlctsxReturn = true
		}
		if (*i) == (*mplusn) {
			(*fs) = false
			(*i) = 0
		}
	} else {
		(*i) = (*i) + 1
		if (*i) <= (*n) {
			zlctsxReturn = true
		} else {
			zlctsxReturn = false
		}
		if (*i) == (*mplusn) {
			(*fs) = true
			(*i) = 0
		}
	}

	return
}
