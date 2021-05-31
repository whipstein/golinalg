package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrec tests the error exits for the routines for eigen- condition
// estimation for DOUBLE PRECISION matrices:
//    ZTRSYL, ZTREXC, ZTRSNA and ZTRSEN.
func Zerrec(path []byte, t *testing.T) {
	var one, scale, zero float64
	var i, ifst, ilst, info, j, lw, m, nmax, nt int
	nmax = 4
	lw = nmax * (nmax + 2)
	one = 1.0
	zero = 0.0
	sel := make([]bool, 4)
	work := cvf(lw)
	x := cvf(4)
	rw := vf(lw)
	s := vf(4)
	sep := vf(4)
	a := cmf(4, 4, opts)
	b := cmf(4, 4, opts)
	c := cmf(4, 4, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	*ok = true
	nt = 0

	//     Initialize A, B and SEL
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, zero)
			b.SetRe(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.SetRe(i-1, i-1, one)
		sel[i-1] = true
	}

	//     Test ZTRSYL
	*srnamt = "ZTRSYL"
	*infot = 1
	golapack.Ztrsyl('X', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 2
	golapack.Ztrsyl('N', 'X', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 3
	golapack.Ztrsyl('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 4
	golapack.Ztrsyl('N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 5
	golapack.Ztrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 7
	golapack.Ztrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 2; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 9
	golapack.Ztrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	*infot = 11
	golapack.Ztrsyl('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), &scale, &info)
	Chkxer("ZTRSYL", &info, lerr, ok, t)
	nt = nt + 8

	//     Test ZTREXC
	*srnamt = "ZTREXC"
	ifst = 1
	ilst = 1
	*infot = 1
	golapack.Ztrexc('X', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 2
	golapack.Ztrexc('N', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 4
	ilst = 2
	golapack.Ztrexc('N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 6
	golapack.Ztrexc('V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 7
	ifst = 0
	ilst = 1
	golapack.Ztrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 7
	ifst = 2
	golapack.Ztrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 8
	ifst = 1
	ilst = 0
	golapack.Ztrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	*infot = 8
	ilst = 2
	golapack.Ztrexc('V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
	Chkxer("ZTREXC", &info, lerr, ok, t)
	nt = nt + 8

	//     Test ZTRSNA
	*srnamt = "ZTRSNA"
	*infot = 1
	golapack.Ztrsna('X', 'A', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 2
	golapack.Ztrsna('B', 'X', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 4
	golapack.Ztrsna('B', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 6
	golapack.Ztrsna('V', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.CMatrix(2, opts), func() *int { y := 2; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 8
	golapack.Ztrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.CMatrix(2, opts), func() *int { y := 2; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 10
	golapack.Ztrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.CMatrix(2, opts), func() *int { y := 2; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 13
	golapack.Ztrsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), s, sep, func() *int { y := 0; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 13
	golapack.Ztrsna('B', 'S', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 1; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	*infot = 16
	golapack.Ztrsna('B', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), c, func() *int { y := 2; return &y }(), s, sep, func() *int { y := 2; return &y }(), &m, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), rw, &info)
	Chkxer("ZTRSNA", &info, lerr, ok, t)
	nt = nt + 9

	//     Test ZTRSEN
	sel[0] = false
	*srnamt = "ZTRSEN"
	*infot = 1
	golapack.Ztrsen('X', 'N', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 2
	golapack.Ztrsen('N', 'X', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 4
	golapack.Ztrsen('N', 'N', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 6
	golapack.Ztrsen('N', 'N', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 8
	golapack.Ztrsen('N', 'V', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 14
	golapack.Ztrsen('N', 'V', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 0; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 14
	golapack.Ztrsen('E', 'V', sel, func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), b, func() *int { y := 3; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	*infot = 14
	golapack.Ztrsen('V', 'V', sel, func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), b, func() *int { y := 3; return &y }(), x, &m, s.GetPtr(0), sep.GetPtr(0), work, func() *int { y := 3; return &y }(), &info)
	Chkxer("ZTRSEN", &info, lerr, ok, t)
	nt = nt + 8

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
