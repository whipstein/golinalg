package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrec tests the error exits for the routines for eigen- condition
// estimation for DOUBLE PRECISION matrices:
//    Dtrsyl, Dtrexc, Dtrsna and Dtrsen.
func derrec(path string, t *testing.T) {
	var one, zero float64
	var i, ifst, ilst, j, nmax, nt int
	var err error

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

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
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

	//     Test Dtrsyl
	*srnamt = "Dtrsyl"
	*errt = fmt.Errorf("!notrna && trana != Trans && trana != ConjTrans: trana=Unrecognized: X")
	_, _, err = golapack.Dtrsyl('X', NoTrans, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("!notrnb && tranb != Trans && tranb != ConjTrans: tranb=Unrecognized: X")
	_, _, err = golapack.Dtrsyl(NoTrans, 'X', 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("isgn != 1 && isgn != -1: isgn=0")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 1, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(2))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	_, _, err = golapack.Dtrsyl(NoTrans, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1))
	chkxer2("Dtrsyl", err)
	nt = nt + 8

	//     Test Dtrexc
	*srnamt = "Dtrexc"
	ifst = 1
	ilst = 1
	*errt = fmt.Errorf("!wantq && compq != 'N': compq='X'")
	_, _, _, err = golapack.Dtrexc('X', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, _, err = golapack.Dtrexc('N', -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	ilst = 2
	_, _, _, err = golapack.Dtrexc('N', 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < max(1, n)): compq='V', q.Rows=1, n=2")
	_, _, _, err = golapack.Dtrexc('V', 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=0, n=1")
	ifst = 0
	ilst = 1
	_, _, _, err = golapack.Dtrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=2, n=1")
	ifst = 2
	_, _, _, err = golapack.Dtrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=0, n=1")
	ifst = 1
	ilst = 0
	_, _, _, err = golapack.Dtrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	*errt = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=2, n=1")
	ilst = 2
	_, _, _, err = golapack.Dtrexc('V', 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), ifst, ilst, work)
	chkxer2("Dtrexc", err)
	nt = nt + 8

	//     Test Dtrsna
	*srnamt = "Dtrsna"
	*errt = fmt.Errorf("!wants && !wantsp: job='X'")
	_, err = golapack.Dtrsna('X', 'A', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.Matrix(1, opts).UpdateRows(1), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("howmny != 'A' && !somcon: howmny='X'")
	_, err = golapack.Dtrsna('B', 'X', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.Matrix(1, opts).UpdateRows(1), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, err = golapack.Dtrsna('B', 'A', sel, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 1, work.Matrix(1, opts).UpdateRows(1), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	_, err = golapack.Dtrsna('V', 'A', sel, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 2, work.Matrix(1, opts).UpdateRows(2), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("vl.Rows < 1 || (wants && vl.Rows < n): job='B', vl.Rows=1, n=2")
	_, err = golapack.Dtrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(2), s, sep, 2, work.Matrix(1, opts).UpdateRows(2), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("vr.Rows < 1 || (wants && vr.Rows < n): job='B', vr.Rows=1, n=2")
	_, err = golapack.Dtrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), s, sep, 2, work.Matrix(1, opts).UpdateRows(2), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("mm < m: mm=0, m=1")
	_, err = golapack.Dtrsna('B', 'A', sel, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), s, sep, 0, work.Matrix(1, opts).UpdateRows(1), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("mm < m: mm=1, m=2")
	_, err = golapack.Dtrsna('B', 'S', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(2), s, sep, 1, work.Matrix(1, opts).UpdateRows(2), &iwork)
	chkxer2("Dtrsna", err)
	*errt = fmt.Errorf("work.Rows < 1 || (wantsp && work.Rows < n): job='B', work.Rows=1, n=2")
	_, err = golapack.Dtrsna('B', 'A', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(2), s, sep, 2, work.Matrix(1, opts).UpdateRows(1), &iwork)
	chkxer2("Dtrsna", err)
	nt = nt + 9

	//     Test Dtrsen
	sel[0] = false
	*srnamt = "Dtrsen"
	*errt = fmt.Errorf("job != 'N' && !wants && !wantsp: job='X'")
	_, _, _, _, err = golapack.Dtrsen('X', 'N', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("compq != 'N' && !wantq: compq='X'")
	_, _, _, _, err = golapack.Dtrsen('N', 'X', sel, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, _, _, err = golapack.Dtrsen('N', 'N', sel, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	_, _, _, _, err = golapack.Dtrsen('N', 'N', sel, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), wr, wi, s.Get(0), sep.Get(0), work, 2, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): compq='V', q.Rows=1, n=2")
	_, _, _, _, err = golapack.Dtrsen('N', 'V', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=0, lwmin=2, lquery=false")
	_, _, _, _, err = golapack.Dtrsen('N', 'V', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), wr, wi, s.Get(0), sep.Get(0), work, 0, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=1, lwmin=2, lquery=false")
	_, _, _, _, err = golapack.Dtrsen('E', 'V', sel, 3, a.Off(0, 0).UpdateRows(3), b.Off(0, 0).UpdateRows(3), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 1)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("lwork < lwmin && !lquery: lwork=3, lwmin=4, lquery=false")
	_, _, _, _, err = golapack.Dtrsen('V', 'V', sel, 3, a.Off(0, 0).UpdateRows(3), b.Off(0, 0).UpdateRows(3), wr, wi, s.Get(0), sep.Get(0), work, 3, &iwork, 2)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("liwork < liwmin && !lquery: liwork=0, liwmin=1, lquery=false")
	_, _, _, _, err = golapack.Dtrsen('E', 'V', sel, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), wr, wi, s.Get(0), sep.Get(0), work, 1, &iwork, 0)
	chkxer2("Dtrsen", err)
	*errt = fmt.Errorf("liwork < liwmin && !lquery: liwork=1, liwmin=2, lquery=false")
	_, _, _, _, err = golapack.Dtrsen('V', 'V', sel, 3, a.Off(0, 0).UpdateRows(3), b.Off(0, 0).UpdateRows(3), wr, wi, s.Get(0), sep.Get(0), work, 4, &iwork, 1)
	chkxer2("Dtrsen", err)
	nt = nt + 10

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}

	if !(*ok) {
		t.Fail()
	}
}
