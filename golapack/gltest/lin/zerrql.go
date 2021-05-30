package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrql tests the error exits for the COMPLEX*16 routines
// that use the QL decomposition of a general matrix.
func Zerrql(path []byte, t *testing.T) {
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

	//     Error exits for QL factorization
	//
	//     ZGEQLF
	*srnamt = "ZGEQLF"
	*infot = 1
	golapack.Zgeqlf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLF", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeqlf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLF", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeqlf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLF", &info, lerr, ok, t)
	*infot = 7
	golapack.Zgeqlf(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLF", &info, lerr, ok, t)

	//     ZGEQL2
	*srnamt = "ZGEQL2"
	*infot = 1
	golapack.Zgeql2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQL2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zgeql2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQL2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zgeql2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("ZGEQL2", &info, lerr, ok, t)

	//     ZGEQLS
	*srnamt = "ZGEQLS"
	*infot = 1
	Zgeqls(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 2
	Zgeqls(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 2
	Zgeqls(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 3
	Zgeqls(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 5
	Zgeqls(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 8
	Zgeqls(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)
	*infot = 10
	Zgeqls(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQLS", &info, lerr, ok, t)

	//     ZUNGQL
	*srnamt = "ZUNGQL"
	*infot = 1
	golapack.Zungql(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungql(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 2
	golapack.Zungql(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungql(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 3
	golapack.Zungql(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 5
	golapack.Zungql(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)
	*infot = 8
	golapack.Zungql(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNGQL", &info, lerr, ok, t)

	//     ZUNG2L
	*srnamt = "ZUNG2L"
	*infot = 1
	golapack.Zung2l(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)
	*infot = 2
	golapack.Zung2l(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)
	*infot = 2
	golapack.Zung2l(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)
	*infot = 3
	golapack.Zung2l(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)
	*infot = 3
	golapack.Zung2l(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)
	*infot = 5
	golapack.Zung2l(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("ZUNG2L", &info, lerr, ok, t)

	//     ZUNMQL
	*srnamt = "ZUNMQL"
	*infot = 1
	golapack.Zunmql('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunmql('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunmql('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunmql('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmql('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmql('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunmql('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmql('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunmql('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunmql('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmql('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)
	*infot = 12
	golapack.Zunmql('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZUNMQL", &info, lerr, ok, t)

	//     ZUNM2L
	*srnamt = "ZUNM2L"
	*infot = 1
	golapack.Zunm2l('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 2
	golapack.Zunm2l('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 3
	golapack.Zunm2l('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 4
	golapack.Zunm2l('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2l('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2l('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 5
	golapack.Zunm2l('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunm2l('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 7
	golapack.Zunm2l('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)
	*infot = 10
	golapack.Zunm2l('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZUNM2L", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
