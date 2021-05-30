package eig

import "golinalg/golapack/gltest"

// Dlctsx This function is used to determine what eigenvalues will be
// selected.  If this is part of the test driver DDRGSX, do not
// change the code UNLESS you are testing input examples and not
// using the built-in examples.
func Dlctsx(ar, ai, beta *float64) (dlctsxReturn bool) {
	m := &gltest.Common.Mn.M
	n := &gltest.Common.Mn.N
	mplusn := &gltest.Common.Mn.Mplusn
	i := &gltest.Common.Mn.I
	fs := &gltest.Common.Mn.Fs

	if *fs {
		(*i) = (*i) + 1
		if (*i) <= (*m) {
			dlctsxReturn = false
		} else {
			dlctsxReturn = true
		}
		if (*i) == (*mplusn) {
			(*fs) = false
			(*i) = 0
		}
	} else {
		(*i) = (*i) + 1
		if (*i) <= (*n) {
			dlctsxReturn = true
		} else {
			dlctsxReturn = false
		}
		if (*i) == (*mplusn) {
			(*fs) = true
			(*i) = 0
		}
	}

	//       IF( AR/BETA.GT.0.0 )THEN
	//          DLCTSX = .TRUE.
	//       ELSE
	//          DLCTSX = .FALSE.
	//       END IF
	return
}
