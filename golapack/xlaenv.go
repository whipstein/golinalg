package golapack

// Xlaenv sets certain machine- and problem-dependent quantities
// which will later be retrieved by ILAENV.
func Xlaenv(ispec, nvalue int) {
	iparms := &common.claenv.iparms

	if ispec >= 1 && ispec <= 9 {
		(*iparms)[ispec-1] = nvalue
	}
}
