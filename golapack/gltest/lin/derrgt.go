package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrgt tests the error exits for the DOUBLE PRECISION tridiagonal
// routines.
func Derrgt(path []byte, t *testing.T) {
	var anorm, rcond float64
	var info int

	b := mf(2, 1, opts)
	c := vf(2)
	cf := vf(2)
	d := vf(2)
	df := vf(2)
	e := vf(2)
	ef := vf(2)
	f := vf(2)
	r1 := vf(2)
	r2 := vf(2)
	w := vf(2)
	x := mf(2, 1, opts)
	ip := make([]int, 2)
	iw := make([]int, 2)
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	d.Set(0, 1.)
	d.Set(1, 2.)
	df.Set(0, 1.)
	df.Set(1, 2.)
	e.Set(0, 3.)
	e.Set(1, 4.)
	ef.Set(0, 3.)
	ef.Set(1, 4.)
	anorm = 1.0
	(*ok) = true

	if string(c2) == "GT" {
		//        Test error exits for the general tridiagonal routines.
		//
		//        DGTTRF
		*srnamt = "DGTTRF"
		*infot = 1
		golapack.Dgttrf(toPtr(-1), c, d, e, f, &ip, &info)
		if *infot == absint(info) {
			*lerr = true
		}
		Chkxer("DGTTRF", &info, lerr, ok, t)

		//        DGTTRS
		*srnamt = "DGTTRS"
		*infot = 1
		golapack.Dgttrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), c, d, e, f, &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGTTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgttrs('N', toPtr(-1), func() *int { y := 0; return &y }(), c, d, e, f, &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGTTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgttrs('N', func() *int { y := 0; return &y }(), toPtr(-1), c, d, e, f, &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGTTRS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgttrs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), c, d, e, f, &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGTTRS", &info, lerr, ok, t)

		//        DGTRFS
		*srnamt = "DGTRFS"
		*infot = 1
		golapack.Dgtrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), c, d, e, cf, df, ef, f, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DGTRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgtrfs('N', toPtr(-1), func() *int { y := 0; return &y }(), c, d, e, cf, df, ef, f, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DGTRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgtrfs('N', func() *int { y := 0; return &y }(), toPtr(-1), c, d, e, cf, df, ef, f, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DGTRFS", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgtrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), c, d, e, cf, df, ef, f, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DGTRFS", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgtrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), c, d, e, cf, df, ef, f, &ip, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DGTRFS", &info, lerr, ok, t)

		//        DGTCON
		*srnamt = "DGTCON"
		*infot = 1
		golapack.Dgtcon('/', func() *int { y := 0; return &y }(), c, d, e, f, &ip, &anorm, &rcond, w, &iw, &info)
		Chkxer("DGTCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgtcon('I', toPtr(-1), c, d, e, f, &ip, &anorm, &rcond, w, &iw, &info)
		Chkxer("DGTCON", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgtcon('I', func() *int { y := 0; return &y }(), c, d, e, f, &ip, func() *float64 { y := -anorm; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DGTCON", &info, lerr, ok, t)
		//
	} else if string(c2) == "PT" {
		//        Test error exits for the positive definite tridiagonal
		//        routines.
		//
		//        DPTTRF
		*srnamt = "DPTTRF"
		*infot = 1
		golapack.Dpttrf(toPtr(-1), d, e, &info)
		Chkxer("DPTTRF", &info, lerr, ok, t)

		//        DPTTRS
		*srnamt = "DPTTRS"
		*infot = 1
		golapack.Dpttrs(toPtr(-1), func() *int { y := 0; return &y }(), d, e, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPTTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpttrs(func() *int { y := 0; return &y }(), toPtr(-1), d, e, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPTTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpttrs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPTTRS", &info, lerr, ok, t)

		//        DPTRFS
		*srnamt = "DPTRFS"
		*infot = 1
		golapack.Dptrfs(toPtr(-1), func() *int { y := 0; return &y }(), d, e, df, ef, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &info)
		Chkxer("DPTRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dptrfs(func() *int { y := 0; return &y }(), toPtr(-1), d, e, df, ef, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &info)
		Chkxer("DPTRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dptrfs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, df, ef, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &info)
		Chkxer("DPTRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dptrfs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, df, ef, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &info)
		Chkxer("DPTRFS", &info, lerr, ok, t)

		//        DPTCON
		*srnamt = "DPTCON"
		*infot = 1
		golapack.Dptcon(toPtr(-1), d, e, &anorm, &rcond, w, &info)
		Chkxer("DPTCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dptcon(func() *int { y := 0; return &y }(), d, e, func() *float64 { y := -anorm; return &y }(), &rcond, w, &info)
		Chkxer("DPTCON", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
