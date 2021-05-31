package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrgt tests the error exits for the COMPLEX*16 tridiagonal
// routines.
func Zerrgt(path []byte, t *testing.T) {
	var anorm, rcond float64
	var i, info, nmax int

	b := cvf(2)
	dl := cvf(2)
	dlf := cvf(2)
	du := cvf(2)
	du2 := cvf(2)
	duf := cvf(2)
	e := cvf(2)
	ef := cvf(2)
	w := cvf(2)
	x := cvf(2)
	d := vf(2)
	df := vf(2)
	r1 := vf(2)
	r2 := vf(2)
	rw := vf(2)
	ip := make([]int, 2)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]
	for i = 1; i <= nmax; i++ {
		d.Set(i-1, 1.)
		e.Set(i-1, 2.)
		dl.Set(i-1, 3.)
		du.Set(i-1, 4.)
	}
	anorm = 1.0
	*ok = true

	if string(c2) == "GT" {
		//        Test error exits for the general tridiagonal routines.
		//
		//        ZGTTRF
		*srnamt = "ZGTTRF"
		(*infot) = 1
		golapack.Zgttrf(toPtr(-1), dl, e, du, du2, &ip, &info)
		Chkxer("ZGTTRF", &info, lerr, ok, t)

		//        ZGTTRS
		*srnamt = "ZGTTRS"
		(*infot) = 1
		golapack.Zgttrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), dl, e, du, du2, &ip, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTTRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgttrs('N', toPtr(-1), func() *int { y := 0; return &y }(), dl, e, du, du2, &ip, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTTRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgttrs('N', func() *int { y := 0; return &y }(), toPtr(-1), dl, e, du, du2, &ip, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTTRS", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgttrs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), dl, e, du, du2, &ip, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTTRS", &info, lerr, ok, t)

		//        ZGTRFS
		*srnamt = "ZGTRFS"
		(*infot) = 1
		golapack.Zgtrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZGTRFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgtrfs('N', toPtr(-1), func() *int { y := 0; return &y }(), dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZGTRFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgtrfs('N', func() *int { y := 0; return &y }(), toPtr(-1), dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZGTRFS", &info, lerr, ok, t)
		(*infot) = 13
		golapack.Zgtrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZGTRFS", &info, lerr, ok, t)
		(*infot) = 15
		golapack.Zgtrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZGTRFS", &info, lerr, ok, t)

		//        ZGTCON
		*srnamt = "ZGTCON"
		(*infot) = 1
		golapack.Zgtcon('/', func() *int { y := 0; return &y }(), dl, e, du, du2, &ip, &anorm, &rcond, w, &info)
		Chkxer("ZGTCON", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgtcon('I', toPtr(-1), dl, e, du, du2, &ip, &anorm, &rcond, w, &info)
		Chkxer("ZGTCON", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgtcon('I', func() *int { y := 0; return &y }(), dl, e, du, du2, &ip, toPtrf64(-anorm), &rcond, w, &info)
		Chkxer("ZGTCON", &info, lerr, ok, t)

	} else if string(c2) == "PT" {
		//        Test error exits for the positive definite tridiagonal
		//        routines.
		//
		//        ZPTTRF
		*srnamt = "ZPTTRF"
		(*infot) = 1
		golapack.Zpttrf(toPtr(-1), d, e, &info)
		Chkxer("ZPTTRF", &info, lerr, ok, t)

		//        ZPTTRS
		*srnamt = "ZPTTRS"
		(*infot) = 1
		golapack.Zpttrs('/', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), d, e, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTTRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zpttrs('U', toPtr(-1), func() *int { y := 0; return &y }(), d, e, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTTRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zpttrs('U', func() *int { y := 0; return &y }(), toPtr(-1), d, e, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTTRS", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zpttrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTTRS", &info, lerr, ok, t)

		//        ZPTRFS
		*srnamt = "ZPTRFS"
		(*infot) = 1
		golapack.Zptrfs('/', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), d, e, df, ef, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZPTRFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zptrfs('U', toPtr(-1), func() *int { y := 0; return &y }(), d, e, df, ef, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZPTRFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zptrfs('U', func() *int { y := 0; return &y }(), toPtr(-1), d, e, df, ef, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZPTRFS", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zptrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, df, ef, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZPTRFS", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zptrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), d, e, df, ef, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZPTRFS", &info, lerr, ok, t)

		//        ZPTCON
		*srnamt = "ZPTCON"
		(*infot) = 1
		golapack.Zptcon(toPtr(-1), d, e, &anorm, &rcond, rw, &info)
		Chkxer("ZPTCON", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zptcon(func() *int { y := 0; return &y }(), d, e, toPtrf64(-anorm), &rcond, rw, &info)
		Chkxer("ZPTCON", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
