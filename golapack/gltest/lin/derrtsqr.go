package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrtsqr tests the error exits for the DOUBLE PRECISION routines
// that use the TSQR decomposition of a general matrix.
func Derrtsqr(path []byte, _t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	c := mf(2, 2, opts)
	t := mf(2, 2, opts)
	w := vf(2)
	tau := vf(3)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			c.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for TS factorization
	//
	//     DGEQR
	*srnamt = "DGEQR"
	*infot = 1
	golapack.Dgeqr(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQR", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgeqr(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQR", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgeqr(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQR", &info, lerr, ok, _t)
	*infot = 6
	golapack.Dgeqr(func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQR", &info, lerr, ok, _t)
	*infot = 8
	golapack.Dgeqr(func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 7; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("DGEQR", &info, lerr, ok, _t)

	//     DGEMQR
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "DGEMQR"
	*infot = 1
	golapack.Dgemqr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgemqr('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dgemqr('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgemqr('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemqr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemqr('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 7
	golapack.Dgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dgemqr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 11
	golapack.Dgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)
	*infot = 13
	golapack.Dgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("DGEMQR", &info, lerr, ok, _t)

	//     DGELQ
	*srnamt = "DGELQ"
	*infot = 1
	golapack.Dgelq(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQ", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgelq(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQ", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgelq(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQ", &info, lerr, ok, _t)
	*infot = 6
	golapack.Dgelq(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQ", &info, lerr, ok, _t)
	*infot = 8
	golapack.Dgelq(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 7; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("DGELQ", &info, lerr, ok, _t)

	//     DGEMLQ
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "DGEMLQ"
	*infot = 1
	golapack.Dgemlq('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgemlq('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dgemlq('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgemlq('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemlq('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 7
	golapack.Dgemlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dgemlq('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dgemlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 11
	golapack.Dgemlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)
	*infot = 13
	golapack.Dgemlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("DGEMLQ", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
