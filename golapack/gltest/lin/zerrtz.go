package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrtz tests the error exits for ZTZRZF.
func Zerrtz(path []byte, t *testing.T) {
	var info int

	tau := cvf(2)
	w := cvf(2)
	a := cmf(2, 2, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	a.Set(0, 0, complex(1, -1.))
	a.Set(0, 1, complex(2, -2.))
	a.Set(1, 1, complex(3, -3.))
	a.Set(1, 0, complex(4, -4.))
	w.Set(0, complex(0, 0))
	w.Set(1, complex(0, 0))
	(*ok) = true

	//     Test error exits for the trapezoidal routines.
	if string(c2) == "TZ" {
		//        ZTZRZF
		*srnamt = "ZTZRZF"
		*infot = 1
		golapack.Ztzrzf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTZRZF", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztzrzf(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTZRZF", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztzrzf(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTZRZF", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztzrzf(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZTZRZF", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztzrzf(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTZRZF", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
