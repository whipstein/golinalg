package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrlq tests the error exits for the COMPLEX*16 routines
// that use the LQ decomposition of a general matrix.
func Zerrlq(path []byte, t *testing.T) {
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

	//     Error exits for LQ factorization
	//
	//     ZGELQF
	*srnamt = "ZGELQF"
	*infot = 1
	golapack.Zgelqf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQF", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgelqf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQF", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgelqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZGELQF", &info, lerr, ok, t)
	*infot = 7
	golapack.Zgelqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQF", &info, lerr, ok, t)

	//     ZGELQ2
	*srnamt = "ZGELQ2"
	*infot = 1
	golapack.Zgelq2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGELQ2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgelq2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGELQ2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgelq2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGELQ2", &info, lerr, ok, t)

	//     ZGELQS
	*srnamt = "ZGELQS"
	*infot = 1
	Zgelqs(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 2
	Zgelqs(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 2
	Zgelqs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 3
	Zgelqs(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 5
	Zgelqs(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 8
	Zgelqs(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)
	*infot = 10
	Zgelqs(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQS", &info, lerr, ok, t)

	//     ZUNGLQ
	*srnamt = "ZUNGLQ"
	*infot = 1
	golapack.Zunglq(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunglq(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunglq(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunglq(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunglq(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunglq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)
	*infot = 8
	golapack.Zunglq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGLQ", &info, lerr, ok, t)

	//     ZUNGL2
	*srnamt = "ZUNGL2"
	*infot = 1
	golapack.Zungl2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungl2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungl2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungl2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungl2(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zungl2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNGL2", &info, lerr, ok, t)

	//     ZUNMLQ
	*srnamt = "ZUNMLQ"
	*infot = 1
	golapack.Zunmlq('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunmlq('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunmlq('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunmlq('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmlq('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmlq('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunmlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmlq('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMLQ", &info, lerr, ok, t)

	//     ZUNML2
	*srnamt = "ZUNML2"
	*infot = 1
	golapack.Zunml2('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunml2('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunml2('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunml2('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunml2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunml2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunml2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunml2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunml2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunml2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNML2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
