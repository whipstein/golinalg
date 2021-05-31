package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrqr tests the error exits for the COMPLEX*16 routines
// that use the QR decomposition of a general matrix.
func Zerrqr(path []byte, t *testing.T) {
	var i, info, j, nmax int

	b := cvf(2)
	w := cvf(2)
	x := cvf(2)
	a := cmf(2, 2, opts)
	af := cmf(2, 2, opts)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
			af.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
		}
		b.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QR factorization
	//
	//     ZGEQRF
	*srnamt = "ZGEQRF"
	*infot = 1
	golapack.Zgeqrf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRF", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeqrf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRF", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeqrf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRF", &info, lerr, ok, t)
	*infot = 7
	golapack.Zgeqrf(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRF", &info, lerr, ok, t)

	//     ZGEQRFP
	*srnamt = "ZGEQRFP"
	*infot = 1
	golapack.Zgeqrfp(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRFP", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeqrfp(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRFP", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeqrfp(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRFP", &info, lerr, ok, t)
	*infot = 7
	golapack.Zgeqrfp(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRFP", &info, lerr, ok, t)

	//     ZGEQR2
	*srnamt = "ZGEQR2"
	*infot = 1
	golapack.Zgeqr2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeqr2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeqr2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2", &info, lerr, ok, t)

	//     ZGEQR2P
	*srnamt = "ZGEQR2P"
	*infot = 1
	golapack.Zgeqr2p(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2P", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeqr2p(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2P", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeqr2p(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQR2P", &info, lerr, ok, t)

	//     ZGEQRS
	*srnamt = "ZGEQRS"
	*infot = 1
	Zgeqrs(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 2
	Zgeqrs(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 2
	Zgeqrs(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 3
	Zgeqrs(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 5
	Zgeqrs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 8
	Zgeqrs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)
	*infot = 10
	Zgeqrs(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQRS", &info, lerr, ok, t)

	//     ZUNGQR
	*srnamt = "ZUNGQR"
	*infot = 1
	golapack.Zungqr(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungqr(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungqr(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungqr(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungqr(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Zungqr(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)
	*infot = 8
	golapack.Zungqr(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQR", &info, lerr, ok, t)

	//     ZUNG2R
	*srnamt = "ZUNG2R"
	*infot = 1
	golapack.Zung2r(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Zung2r(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Zung2r(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Zung2r(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Zung2r(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Zung2r(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2R", &info, lerr, ok, t)

	//     ZUNMQR
	*srnamt = "ZUNMQR"
	*infot = 1
	golapack.Zunmqr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunmqr('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunmqr('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunmqr('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmqr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmqr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmqr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmqr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunmqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmqr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmqr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQR", &info, lerr, ok, t)

	//     ZUNM2R
	*srnamt = "ZUNM2R"
	*infot = 1
	golapack.Zunm2r('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunm2r('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunm2r('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunm2r('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2r('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2r('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2r('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunm2r('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunm2r('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunm2r('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2R", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
