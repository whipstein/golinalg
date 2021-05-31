package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrlq tests the error exits for the DOUBLE PRECISION routines
// that use the LQ decomposition of a general matrix.
func Derrlq(path []byte, t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	af := mf(2, 2, opts)
	b := vf(2)
	w := vf(2)
	x := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for LQ factorization
	//
	//     DGELQF
	*srnamt = "DGELQF"
	*infot = 1
	golapack.Dgelqf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQF", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgelqf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQF", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgelqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DGELQF", &info, lerr, ok, t)
	*infot = 7
	golapack.Dgelqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQF", &info, lerr, ok, t)

	//     DGELQ2
	*srnamt = "DGELQ2"
	*infot = 1
	golapack.Dgelq2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGELQ2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgelq2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGELQ2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgelq2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGELQ2", &info, lerr, ok, t)

	//     DGELQS
	*srnamt = "DGELQS"
	*infot = 1
	Dgelqs(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 2
	Dgelqs(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 2
	Dgelqs(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 3
	Dgelqs(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 5
	Dgelqs(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 8
	Dgelqs(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)
	*infot = 10
	Dgelqs(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQS", &info, lerr, ok, t)

	//     DORGLQ
	*srnamt = "DORGLQ"
	*infot = 1
	golapack.Dorglq(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorglq(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorglq(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorglq(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorglq(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorglq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)
	*infot = 8
	golapack.Dorglq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGLQ", &info, lerr, ok, t)

	//     DORGL2
	*srnamt = "DORGL2"
	*infot = 1
	golapack.Dorgl2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgl2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgl2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgl2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgl2(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorgl2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGL2", &info, lerr, ok, t)

	//     DORMLQ
	*srnamt = "DORMLQ"
	*infot = 1
	golapack.Dormlq('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dormlq('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dormlq('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 4
	golapack.Dormlq('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormlq('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormlq('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 10
	golapack.Dormlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormlq('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORMLQ", &info, lerr, ok, t)

	//     DORML2
	*srnamt = "DORML2"
	*infot = 1
	golapack.Dorml2('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorml2('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorml2('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dorml2('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorml2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorml2('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorml2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 7
	golapack.Dorml2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 2; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 7
	golapack.Dorml2('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)
	*infot = 10
	golapack.Dorml2('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, af, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DORML2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
