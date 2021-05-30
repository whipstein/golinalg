package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrbd tests the error exits for ZGEBRD, ZUNGBR, ZUNMBR, and ZBDSQR.
func Zerrbd(path []byte, t *testing.T) {
	var i, info, j, lw, nmax, nt int

	nmax = 4
	lw = nmax
	tp := cvf(4)
	tq := cvf(4)
	w := cvf(lw)
	d := vf(4)
	e := vf(4)
	rw := vf(lw)
	a := cmf(4, 4, opts)
	u := cmf(4, 4, opts)
	v := cmf(4, 4, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))
		}
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the SVD routines.
	if string(c2) == "BD" {
		//        ZGEBRD
		*srnamt = "ZGEBRD"
		*infot = 1
		golapack.Zgebrd(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgebrd(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgebrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGEBRD", &info, lerr, ok, t)
		*infot = 10
		golapack.Zgebrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBRD", &info, lerr, ok, t)
		nt = nt + 4

		//        ZUNGBR
		*srnamt = "ZUNGBR"
		*infot = 1
		golapack.Zungbr('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zungbr('Q', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zungbr('Q', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zungbr('Q', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zungbr('Q', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zungbr('P', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zungbr('P', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zungbr('Q', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zungbr('Q', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		*infot = 9
		golapack.Zungbr('Q', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGBR", &info, lerr, ok, t)
		nt = nt + 10

		//        ZUNMBR
		*srnamt = "ZUNMBR"
		*infot = 1
		golapack.Zunmbr('/', 'L', 'T', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zunmbr('Q', '/', 'T', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zunmbr('Q', 'L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zunmbr('Q', 'L', 'C', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 5
		golapack.Zunmbr('Q', 'L', 'C', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zunmbr('Q', 'L', 'C', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Zunmbr('Q', 'L', 'C', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Zunmbr('Q', 'R', 'C', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Zunmbr('P', 'L', 'C', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Zunmbr('P', 'R', 'C', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 11
		golapack.Zunmbr('Q', 'R', 'C', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 13
		golapack.Zunmbr('Q', 'L', 'C', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		*infot = 13
		golapack.Zunmbr('Q', 'R', 'C', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZUNMBR", &info, lerr, ok, t)
		nt = nt + 13

		//        ZBDSQR
		*srnamt = "ZBDSQR"
		*infot = 1
		golapack.Zbdsqr('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zbdsqr('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zbdsqr('U', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 5
		golapack.Zbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 9
		golapack.Zbdsqr('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 11
		golapack.Zbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		*infot = 13
		golapack.Zbdsqr('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZBDSQR", &info, lerr, ok, t)
		nt = nt + 8
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
