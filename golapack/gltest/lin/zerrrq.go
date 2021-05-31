package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrrq tests the error exits for the COMPLEX*16 routines
// that use the RQ decomposition of a general matrix.
func Zerrrq(path []byte, t *testing.T) {
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

	//     Error exits for RQ factorization
	//
	//     ZGERQF
	*srnamt = "ZGERQF"
	*infot = 1
	golapack.Zgerqf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQF", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgerqf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQF", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgerqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZGERQF", &info, lerr, ok, t)
	*infot = 7
	golapack.Zgerqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQF", &info, lerr, ok, t)

	//     ZGERQ2
	*srnamt = "ZGERQ2"
	*infot = 1
	golapack.Zgerq2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGERQ2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgerq2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGERQ2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgerq2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGERQ2", &info, lerr, ok, t)

	//     ZGERQS
	*srnamt = "ZGERQS"
	*infot = 1
	Zgerqs(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 2
	Zgerqs(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 2
	Zgerqs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 3
	Zgerqs(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 5
	Zgerqs(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 8
	Zgerqs(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)
	*infot = 10
	Zgerqs(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGERQS", &info, lerr, ok, t)

	//     ZUNGRQ
	*srnamt = "ZUNGRQ"
	*infot = 1
	golapack.Zungrq(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungrq(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungrq(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungrq(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungrq(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zungrq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)
	*infot = 8
	golapack.Zungrq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGRQ", &info, lerr, ok, t)

	//     ZUNGR2
	*srnamt = "ZUNGR2"
	*infot = 1
	golapack.Zungr2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungr2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungr2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungr2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungr2(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zungr2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGR2", &info, lerr, ok, t)

	//     ZUNMRQ
	*srnamt = "ZUNMRQ"
	*infot = 1
	golapack.Zunmrq('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunmrq('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunmrq('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunmrq('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmrq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmrq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmrq('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmrq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmrq('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunmrq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmrq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmrq('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMRQ", &info, lerr, ok, t)

	//     ZUNMR2
	*srnamt = "ZUNMR2"
	*infot = 1
	golapack.Zunmr2('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunmr2('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunmr2('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunmr2('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmr2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmr2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmr2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmr2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmr2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunmr2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNMR2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
