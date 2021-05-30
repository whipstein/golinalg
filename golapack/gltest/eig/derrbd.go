package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrbd tests the error exits for DGEBD2, DGEBRD, DORGBR, DORMBR,
// DBDSQR, DBDSDC and DBDSVDX.
func Derrbd(path []byte, t *testing.T) {
	var one, zero float64
	var i, info, j, lw, nmax, ns, nt int

	nmax = 4
	lw = nmax
	zero = 0.0
	one = 1.0
	d := vf(4)
	e := vf(4)
	s := vf(4)
	tp := vf(4)
	tq := vf(4)
	w := vf(lw)
	iw := make([]int, 4)
	a := mf(4, 4, opts)
	q := mf(4, 4, opts)
	u := mf(4, 4, opts)
	v := mf(4, 4, opts)
	iq := make([]int, 4*4)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
		}
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the SVD routines.
	if string(c2) == "BD" {
		//        DGEBRD
		*srnamt = "DGEBRD"
		*infot = 1
		golapack.Dgebrd(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgebrd(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgebrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGEBRD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgebrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), d, e, tq, tp, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBRD", &info, lerr, ok, t)
		nt = nt + 4

		//        DGEBD2
		*srnamt = "DGEBD2"
		*infot = 1
		golapack.Dgebd2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, &info)
		Chkxer("DGEBD2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgebd2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, &info)
		Chkxer("DGEBD2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgebd2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tq, tp, w, &info)
		Chkxer("DGEBD2", &info, lerr, ok, t)
		nt = nt + 3

		//        DORGBR
		*srnamt = "DORGBR"
		*infot = 1
		golapack.Dorgbr('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dorgbr('Q', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorgbr('Q', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorgbr('Q', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorgbr('Q', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorgbr('P', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorgbr('P', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dorgbr('Q', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dorgbr('Q', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		*infot = 9
		golapack.Dorgbr('Q', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tq, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGBR", &info, lerr, ok, t)
		nt = nt + 10

		//        DORMBR
		*srnamt = "DORMBR"
		*infot = 1
		golapack.Dormbr('/', 'L', 'T', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dormbr('Q', '/', 'T', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dormbr('Q', 'L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dormbr('Q', 'L', 'T', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormbr('Q', 'L', 'T', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dormbr('Q', 'L', 'T', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormbr('Q', 'L', 'T', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormbr('Q', 'R', 'T', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormbr('P', 'L', 'T', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormbr('P', 'R', 'T', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 11
		golapack.Dormbr('Q', 'R', 'T', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 13
		golapack.Dormbr('Q', 'L', 'T', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		*infot = 13
		golapack.Dormbr('Q', 'R', 'T', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tq, u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMBR", &info, lerr, ok, t)
		nt = nt + 13

		//        DBDSQR
		*srnamt = "DBDSQR"
		*infot = 1
		golapack.Dbdsqr('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dbdsqr('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dbdsqr('U', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 9
		golapack.Dbdsqr('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 11
		golapack.Dbdsqr('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		*infot = 13
		golapack.Dbdsqr('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), d, e, v, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("DBDSQR", &info, lerr, ok, t)
		nt = nt + 8

		//        DBDSDC
		*srnamt = "DBDSDC"
		*infot = 1
		golapack.Dbdsdc('/', 'N', func() *int { y := 0; return &y }(), d, e, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q.VectorIdx(0), &iq, w, &iw, &info)
		Chkxer("DBDSDC", &info, lerr, ok, t)
		*infot = 2
		golapack.Dbdsdc('U', '/', func() *int { y := 0; return &y }(), d, e, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q.VectorIdx(0), &iq, w, &iw, &info)
		Chkxer("DBDSDC", &info, lerr, ok, t)
		*infot = 3
		golapack.Dbdsdc('U', 'N', toPtr(-1), d, e, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q.VectorIdx(0), &iq, w, &iw, &info)
		Chkxer("DBDSDC", &info, lerr, ok, t)
		*infot = 7
		golapack.Dbdsdc('U', 'I', func() *int { y := 2; return &y }(), d, e, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q.VectorIdx(0), &iq, w, &iw, &info)
		Chkxer("DBDSDC", &info, lerr, ok, t)
		*infot = 9
		golapack.Dbdsdc('U', 'I', func() *int { y := 2; return &y }(), d, e, u, func() *int { y := 2; return &y }(), v, func() *int { y := 1; return &y }(), q.VectorIdx(0), &iq, w, &iw, &info)
		Chkxer("DBDSDC", &info, lerr, ok, t)
		nt = nt + 5

		//        DBDSVDX
		*srnamt = "DBDSVDX"
		*infot = 1
		golapack.Dbdsvdx('X', 'N', 'A', func() *int { y := 1; return &y }(), d, e, &zero, &one, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dbdsvdx('U', 'X', 'A', func() *int { y := 1; return &y }(), d, e, &zero, &one, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dbdsvdx('U', 'V', 'X', func() *int { y := 1; return &y }(), d, e, &zero, &one, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dbdsvdx('U', 'V', 'A', toPtr(-1), d, e, &zero, &one, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dbdsvdx('U', 'V', 'V', func() *int { y := 2; return &y }(), d, e, toPtrf64(-one), &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dbdsvdx('U', 'V', 'V', func() *int { y := 2; return &y }(), d, e, &one, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dbdsvdx('L', 'V', 'I', func() *int { y := 2; return &y }(), d, e, &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dbdsvdx('L', 'V', 'I', func() *int { y := 4; return &y }(), d, e, &zero, &zero, func() *int { y := 5; return &y }(), func() *int { y := 2; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dbdsvdx('L', 'V', 'I', func() *int { y := 4; return &y }(), d, e, &zero, &zero, func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dbdsvdx('L', 'V', 'I', func() *int { y := 4; return &y }(), d, e, &zero, &zero, func() *int { y := 3; return &y }(), func() *int { y := 5; return &y }(), &ns, s, q, func() *int { y := 1; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dbdsvdx('L', 'V', 'A', func() *int { y := 4; return &y }(), d, e, &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 0; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dbdsvdx('L', 'V', 'A', func() *int { y := 4; return &y }(), d, e, &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, q, func() *int { y := 2; return &y }(), w, &iw, &info)
		Chkxer("DBDSVDX", &info, lerr, ok, t)
		nt = nt + 12
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
