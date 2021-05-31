package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrqrt tests the error exits for the COMPLEX*16 routines
// that use the QRT decomposition of a general matrix.
func Zerrqrt(path []byte, _t *testing.T) {
	var i, info, j, nmax int

	w := cvf(2)
	a := cmf(2, 2, opts)
	c := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			c.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			t.Set(i-1, j-1, 1./complex(float64(i+j), 0))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QRT factorization
	//
	//     ZGEQRT
	*srnamt = "ZGEQRT"
	(*infot) = 1
	golapack.Zgeqrt(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEQRT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgeqrt(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEQRT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Zgeqrt(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEQRT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgeqrt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEQRT", &info, lerr, ok, _t)
	(*infot) = 7
	golapack.Zgeqrt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEQRT", &info, lerr, ok, _t)

	//     ZGEQRT2
	*srnamt = "ZGEQRT2"
	(*infot) = 1
	golapack.Zgeqrt2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT2", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgeqrt2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT2", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Zgeqrt2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT2", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Zgeqrt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT2", &info, lerr, ok, _t)

	//     ZGEQRT3
	*srnamt = "ZGEQRT3"
	(*infot) = 1
	golapack.Zgeqrt3(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT3", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgeqrt3(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT3", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Zgeqrt3(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT3", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Zgeqrt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRT3", &info, lerr, ok, _t)

	//     ZGEMQRT
	*srnamt = "ZGEMQRT"
	(*infot) = 1
	golapack.Zgemqrt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgemqrt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Zgemqrt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Zgemqrt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgemqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgemqrt('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Zgemqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 8
	golapack.Zgemqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 8
	golapack.Zgemqrt('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 10
	golapack.Zgemqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)
	(*infot) = 12
	golapack.Zgemqrt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("ZGEMQRT", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
