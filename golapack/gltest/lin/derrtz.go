package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrtz tests the error exits for STZRZF.
func Derrtz(path []byte, t *testing.T) {
	var info int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	a := mf(2, 2, opts)
	tau := vf(2)
	w := vf(2)

	c2 := path[1:3]
	a.Set(0, 0, 1.)
	a.Set(0, 1, 2.)
	a.Set(1, 1, 3.)
	a.Set(1, 0, 4.)
	w.Set(0, 0.0)
	w.Set(1, 0.0)
	(*ok) = true

	if string(c2) == "TZ" {
		//        Test error exits for the trapezoidal routines.
		//
		//        DTZRZF
		*srnamt = "DTZRZF"
		*infot = 1
		golapack.Dtzrzf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTZRZF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtzrzf(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTZRZF", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtzrzf(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTZRZF", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtzrzf(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DTZRZF", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtzrzf(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTZRZF", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
