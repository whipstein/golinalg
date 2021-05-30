package lin

import "golinalg/golapack/gltest"

// Xlaenv sets certain machine- and problem-dependent quantities
// which will later be retrieved by ILAENV.
func Xlaenv(ispec, nvalue int) {
	iparms := &gltest.Common.Claenv.Iparms

	if ispec >= 1 && ispec <= 9 {
		(*iparms)[ispec-1] = nvalue
	}
}
