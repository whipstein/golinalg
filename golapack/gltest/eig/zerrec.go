package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrec tests the error exits for the routines for eigen- condition
// estimation for DOUBLE PRECISION matrices:
//    Ztrsyl, Ztrexc, Ztrsna and Ztrsen.
func zerrec(path string, t *testing.T) (nt int) {
	var one, zero float64
	var i, ifst, ilst, j, lw, nmax int
	var err error

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

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
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

	//     Test Ztrsyl
	*srnamt = "Ztrsyl"
	*errt = fmt.Errorf("!notrna && trana != ConjTrans: trana=Unrecognized: X")
	_, _, err = golapack.Ztrsyl('X', NoTrans, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("!notrnb && tranb != ConjTrans: tranb=Unrecognized: X")
	_, _, err = golapack.Ztrsyl(NoTrans, 'X', 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("isgn != 1 && isgn != -1: isgn=0")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 1, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(2))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	_, _, err = golapack.Ztrsyl(NoTrans, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Ztrsyl", err)
	nt = nt + 8

	//     Test Ztrexc
	*srnamt = "Ztrexc"
	ifst = 1
	ilst = 1
	*errt = fmt.Errorf("compq != 'N' && !wantq: compq='X'")
	err = golapack.Ztrexc('X', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztrexc('N', -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	ilst = 2
	err = golapack.Ztrexc('N', 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < max(1, n)): compq='V', q.Rows=1, n=2")
	err = golapack.Ztrexc('V', 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=0, n=1")
	ifst = 0
	ilst = 1
	err = golapack.Ztrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=2, n=1")
	ifst = 2
	err = golapack.Ztrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=0, n=1")
	ifst = 1
	ilst = 0
	err = golapack.Ztrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	*errt = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=2, n=1")
	ilst = 2
	err = golapack.Ztrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst)
	chkxer2("Ztrexc", err)
	nt = nt + 8

	//     Test Ztrsna
	*srnamt = "Ztrsna"
	*errt = fmt.Errorf("!wants && !wantsp: job='X'")
	_, err = golapack.Ztrsna('X', 'A', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("howmny != 'A' && !somcon: howmny='X'")
	_, err = golapack.Ztrsna('B', 'X', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, err = golapack.Ztrsna('B', 'A', sel, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	_, err = golapack.Ztrsna('V', 'A', sel, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 2, work.CMatrix(2, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("vl.Rows < 1 || (wants && vl.Rows < n): job='B', vl.Rows=1, n=2")
	_, err = golapack.Ztrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(2), s, sep, 2, work.CMatrix(2, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("vr.Rows < 1 || (wants && vr.Rows < n): job='B', vr.Rows=1, n=2")
	_, err = golapack.Ztrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), s, sep, 2, work.CMatrix(2, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("mm < m: mm=0, m=1")
	_, err = golapack.Ztrsna('B', 'A', sel, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 0, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("mm < m: mm=1, m=2")
	_, err = golapack.Ztrsna('B', 'S', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(2), s, sep, 1, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	*errt = fmt.Errorf("work.Rows < 1 || (wantsp && work.Rows < n): job='B', work.Rows=1, n=2")
	_, err = golapack.Ztrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(2), s, sep, 2, work.CMatrix(1, opts), rw)
	chkxer2("Ztrsna", err)
	nt = nt + 9

	//     Test Ztrsen
	sel[0] = false
	*srnamt = "Ztrsen"
	*errt = fmt.Errorf("job != 'N' && !wants && !wantsp: job='X'")
	_, _, _, err = golapack.Ztrsen('X', 'N', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x, work, 1)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("compq != 'N' && !wantq: compq='X'")
	_, _, _, err = golapack.Ztrsen('N', 'X', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x, work, 1)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, _, err = golapack.Ztrsen('N', 'N', sel, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x, work, 1)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	_, _, _, err = golapack.Ztrsen('N', 'N', sel, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x, work, 2)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): compq='V', q.Rows=1, n=2")
	_, _, _, err = golapack.Ztrsen('N', 'V', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), x, work, 1)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=0, lwmin=1, lquery=false")
	_, _, _, err = golapack.Ztrsen('N', 'V', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x, work, 0)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=1, lwmin=2, lquery=false")
	_, _, _, err = golapack.Ztrsen('E', 'V', sel, 3, a.Off(0, 0).UpdateRows(3), b.Off(0, 0).UpdateRows(3), x, work, 1)
	chkxer2("Ztrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=3, lwmin=4, lquery=false")
	_, _, _, err = golapack.Ztrsen('V', 'V', sel, 3, a.Off(0, 0).UpdateRows(3), b.Off(0, 0).UpdateRows(3), x, work, 3)
	chkxer2("Ztrsen", err)
	nt = nt + 8

	//     Print a summary line.
	// if *ok {
	// 	fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	// } else {
	// 	fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	// }
	*srnamt = ""
	if !(*ok) {
		t.Fail()
	}

	return
}
