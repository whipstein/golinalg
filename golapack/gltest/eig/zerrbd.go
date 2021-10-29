package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrbd tests the error exits for Zgebrd, Zungbr, Zunmbr, and Zbdsqr.
func zerrbd(path string, t *testing.T) {
	var i, j, lw, nmax, nt int
	var err error

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

	errt := &gltest.Common.Infoc.Errt
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
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
	if c2 == "bd" {
		//        Zgebrd
		*srnamt = "Zgebrd"
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Zgebrd(-1, 0, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 1)
		chkxer2("Zgebrd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgebrd(0, -1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 1)
		chkxer2("Zgebrd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Zgebrd(2, 1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 2)
		chkxer2("Zgebrd", err)
		*errt = fmt.Errorf("lwork < max(1, m, n) && !lquery: lwork=1, m=2, n=1")
		err = golapack.Zgebrd(2, 1, a.Off(0, 0).UpdateRows(2), d, e, tq, tp, w, 1)
		chkxer2("Zgebrd", err)
		nt = nt + 4

		//        Zungbr
		*srnamt = "Zungbr"
		*errt = fmt.Errorf("!wantq && vect != 'P': vect='/'")
		err = golapack.Zungbr('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Zungbr('Q', -1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', m=0, n=-1, k=0")
		err = golapack.Zungbr('Q', 0, -1, 0, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', m=0, n=1, k=0")
		err = golapack.Zungbr('Q', 0, 1, 0, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', m=1, n=0, k=1")
		err = golapack.Zungbr('Q', 1, 0, 1, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='P', m=1, n=0, k=0")
		err = golapack.Zungbr('P', 1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='P', m=0, n=1, k=1")
		err = golapack.Zungbr('P', 0, 1, 1, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("k < 0: k=-1")
		err = golapack.Zungbr('Q', 0, 0, -1, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Zungbr('Q', 2, 1, 1, a.Off(0, 0).UpdateRows(1), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		*errt = fmt.Errorf("lwork < max(1, mn) && !lquery: lwork=1, mn=2, lquery=false")
		err = golapack.Zungbr('Q', 2, 2, 1, a.Off(0, 0).UpdateRows(2), tq, w.Off(0, 1), 1)
		chkxer2("Zungbr", err)
		nt = nt + 10

		//        Zunmbr
		*srnamt = "Zunmbr"
		*errt = fmt.Errorf("!applyq && vect != 'P': vect='/'")
		err = golapack.Zunmbr('/', Left, Trans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
		err = golapack.Zunmbr('Q', '/', Trans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zunmbr('Q', Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Zunmbr('Q', Left, ConjTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zunmbr('Q', Left, ConjTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("k < 0: k=-1")
		err = golapack.Zunmbr('Q', Left, ConjTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='Q', a.Rows=1, nq=2, k=0")
		err = golapack.Zunmbr('Q', Left, ConjTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='Q', a.Rows=1, nq=2, k=0")
		err = golapack.Zunmbr('Q', Right, ConjTrans, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='P', a.Rows=1, nq=2, k=2")
		err = golapack.Zunmbr('P', Left, ConjTrans, 2, 0, 2, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='P', a.Rows=1, nq=2, k=2")
		err = golapack.Zunmbr('P', Right, ConjTrans, 0, 2, 2, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
		err = golapack.Zunmbr('Q', Right, ConjTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=0, nw=0, lquery=false")
		err = golapack.Zunmbr('Q', Left, ConjTrans, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w.Off(0, 0), 0)
		chkxer2("Zunmbr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=0, nw=0, lquery=false")
		err = golapack.Zunmbr('Q', Right, ConjTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w.Off(0, 0), 0)
		chkxer2("Zunmbr", err)
		nt = nt + 13

		//        Zbdsqr
		*srnamt = "Zbdsqr"
		*errt = fmt.Errorf("uplo != Upper && !lower: uplo=Unrecognized: /")
		_, err = golapack.Zbdsqr('/', 0, 0, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zbdsqr(Upper, -1, 0, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("ncvt < 0: ncvt=-1")
		_, err = golapack.Zbdsqr(Upper, 0, -1, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("nru < 0: nru=-1")
		_, err = golapack.Zbdsqr(Upper, 0, 0, -1, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("ncc < 0: ncc=-1")
		_, err = golapack.Zbdsqr(Upper, 0, 0, 0, -1, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("(ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)): ncvt=1, vt.Rows=1, n=2")
		_, err = golapack.Zbdsqr(Upper, 2, 1, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("u.Rows < max(1, nru): u.Rows=1, nru=2")
		_, err = golapack.Zbdsqr(Upper, 0, 0, 2, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		*errt = fmt.Errorf("(ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)): ncc=1, c.Rows=1, n=2")
		_, err = golapack.Zbdsqr(Upper, 2, 0, 0, 1, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), rw)
		chkxer2("Zbdsqr", err)
		nt = nt + 8
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
	*infot = 0
	*srnamt = ""
	if !(*ok) {
		t.Fail()
	}
}
