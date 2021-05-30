package eig

import (
	"golinalg/golapack/gltest"
	"math/cmplx"
)

// Zslect returns .TRUE. if the eigenvalue Z is to be selected,
// otherwise it returns .FALSE.
// It is used by ZCHK41 to test if ZGEES successfully sorts eigenvalues,
// and by ZCHK43 to test if ZGEESX successfully sorts eigenvalues.
//
// The common block /SSLCT/ controls how eigenvalues are selected.
// If SELOPT = 0, then ZSLECT return .TRUE. when real(Z) is less than
// zero, and .FALSE. otherwise.
// If SELOPT is at least 1, ZSLECT returns SELVAL(SELOPT) and adds 1
// to SELOPT, cycling back to 1 at SELMAX.
func Zslect(z complex128) (zslectReturn bool) {
	var rmin, x, zero float64
	var i int

	zero = 0.0
	selopt := &gltest.Common.Sslct.Selopt
	seldim := &gltest.Common.Sslct.Seldim
	selval := &gltest.Common.Sslct.Selval
	selwi := gltest.Common.Sslct.Selwi
	selwr := gltest.Common.Sslct.Selwr

	if (*selopt) == 0 {
		zslectReturn = (real(z) < zero)
	} else {
		rmin = cmplx.Abs(z - complex(selwr.Get(0), selwi.Get(0)))
		zslectReturn = (*selval)[0]
		for i = 2; i <= (*seldim); i++ {
			x = cmplx.Abs(z - complex(selwr.Get(i-1), selwi.Get(i-1)))
			if x <= rmin {
				rmin = x
				zslectReturn = (*selval)[i-1]
			}
		}
	}
	return
}
