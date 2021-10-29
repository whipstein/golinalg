package eig

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// dslect returns .TRUE. if the eigenvalue ZR+sqrt(-1)*ZI is to be
// selected, and otherwise it returns .FALSE.
// It is used by DCHK41 to test if DGEES successfully sorts eigenvalues,
// and by DCHK43 to test if DGEESX successfully sorts eigenvalues.
//
// The common block /SSLCT/ controls how eigenvalues are selected.
// If SELOPT = 0, then DSLECT return .TRUE. when ZR is less than zero,
// and .FALSE. otherwise.
// If SELOPT is at least 1, DSLECT returns SELVAL(SELOPT) and adds 1
// to SELOPT, cycling back to 1 at SELMAX.
func dslect(zr, zi *float64) (dslectReturn bool) {
	var rmin, x, zero float64
	var i int

	selopt := &gltest.Common.Sslct.Selopt
	seldim := &gltest.Common.Sslct.Seldim
	selval := &gltest.Common.Sslct.Selval
	selwr := gltest.Common.Sslct.Selwr
	selwi := gltest.Common.Sslct.Selwi

	zero = 0.0

	if (*selopt) == 0 {
		dslectReturn = ((*zr) < zero)
	} else {
		rmin = golapack.Dlapy2((*zr)-float64(selwr.Get(0)), (*zi)-float64(selwi.Get(0)))
		dslectReturn = (*selval)[0]
		for i = 2; i <= (*seldim); i++ {
			x = golapack.Dlapy2((*zr)-float64(selwr.Get(i-1)), (*zi)-float64(selwi.Get(i-1)))
			if x <= rmin {
				rmin = x
				dslectReturn = (*selval)[i-1]
			}
		}
	}
	return
}
