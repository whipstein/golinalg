package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrec tests the error exits for the routines for eigen- condition
// estimation for DOUBLE PRECISION matrices:
//    DTRSYL, DTREXC, DTRSNA and DTRSEN.
func Derrec(path []byte, t *testing.T) {
	var one, scale, zero float64
	var i, ifst, ilst, info, j, m, nmax, nt int

	sel := make([]bool, 4)
	s := vf(4)
	sep := vf(4)
	wi := vf(4)
	work := vf(4)
	wr := vf(4)
	iwork := make([]int, 4)
	a := mf(4, 4, opts)
	b := mf(4, 4, opts)
	c := mf(4, 4, opts)

	nmax = 4
	one = 1.0
	zero = 0.0

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	(*ok) = true
	nt = 0

	//     Initialize A, B and SEL
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, zero)
			b.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.Set(i-1, i-1, one)
		sel[i-1] = true
	}

	//     Test DTRSYL
	*srnamt = "DTRSYL"
	*infot = 1
	golapack.Dtrsyl('X', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 2
	golapack.Dtrsyl('N', 'X', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 3
	golapack.Dtrsyl('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 4
	golapack.Dtrsyl('N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 5
	golapack.Dtrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 7
	golapack.Dtrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 2; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 9
	golapack.Dtrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	*infot = 11
	golapack.Dtrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("DTRSYL", &info, lerr, ok, t)
	nt = nt + 8

	//     Test DTREXC
	*srnamt = "DTREXC"
	ifst = 1
	ilst = 1
	*infot = 1
	golapack.Dtrexc('X', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 2
	golapack.Dtrexc('N', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 4
	ilst = 2
	golapack.Dtrexc('N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 6
	golapack.Dtrexc('V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 7
	ifst = 0
	ilst = 1
	golapack.Dtrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 7
	ifst = 2
	golapack.Dtrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 8
	ifst = 1
	ilst = 0
	golapack.Dtrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	*infot = 8
	ilst = 2
	golapack.Dtrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, work, &info)
	Chkxer("DTREXC", &info, lerr, ok, t)
	nt = nt + 8

	//     Test DTRSNA
	*srnamt = "DTRSNA"
	*infot = 1
	golapack.Dtrsna('X', 'A', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 1; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 2
	golapack.Dtrsna('B', 'X', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 1; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 4
	golapack.Dtrsna('B', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 1; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 6
	golapack.Dtrsna('V', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 2; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 8
	golapack.Dtrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 2; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 10
	golapack.Dtrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 2; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 13
	golapack.Dtrsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 0; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 1; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 13
	golapack.Dtrsna('B', 'S', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 2; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	*infot = 16
	golapack.Dtrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.Matrix(1, opts), func() *int { y := 1; return &y }(), &iwork, &info)
	Chkxer("DTRSNA", &info, lerr, ok, t)
	nt = nt + 9

	//     Test DTRSEN
	sel[0] = false
	*srnamt = "DTRSEN"
	*infot = 1
	golapack.Dtrsen('X', 'N', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 2
	golapack.Dtrsen('N', 'X', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 4
	golapack.Dtrsen('N', 'N', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 6
	golapack.Dtrsen('N', 'N', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 2; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 8
	golapack.Dtrsen('N', 'V', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 15
	golapack.Dtrsen('N', 'V', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 0; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 15
	golapack.Dtrsen('E', 'V', sel, func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), b, func() *int { y := 3; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 15
	golapack.Dtrsen('V', 'V', sel, func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), b, func() *int { y := 3; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 3; return &y }(), &iwork, func() *int { y := 2; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 17
	golapack.Dtrsen('E', 'V', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &iwork, func() *int { y := 0; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	*infot = 17
	golapack.Dtrsen('V', 'V', sel, func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), b, func() *int { y := 3; return &y }(), wr, wi, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 4; return &y }(), &iwork, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTRSEN", &info, lerr, ok, t)
	nt = nt + 10

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
