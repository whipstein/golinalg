package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrbd tests the error exits for Dgebd2, Dgebrd, Dorgbr, Dormbr,
// Dbdsqr, Dbdsdc and Dbdsvdx.
func derrbd(path string, t *testing.T) (nt int) {
	var one, zero float64
	var i, j, lw, nmax, ns int
	var err error

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

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
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
	if c2 == "bd" {
		//        Dgebrd
		*srnamt = "Dgebrd"
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dgebrd(-1, 0, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 1)
		chkxer2("Dgebrd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgebrd(0, -1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 1)
		chkxer2("Dgebrd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Dgebrd(2, 1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w, 2)
		chkxer2("Dgebrd", err)
		*errt = fmt.Errorf("lwork < max(1, m, n) && !lquery: lwork=1, m=2, n=1, lquery=false")
		err = golapack.Dgebrd(2, 1, a.Off(0, 0).UpdateRows(2), d, e, tq, tp, w, 1)
		chkxer2("Dgebrd", err)
		nt = nt + 4

		//        Dgebd2
		*srnamt = "Dgebd2"
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dgebd2(-1, 0, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w)
		chkxer2("Dgebd2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgebd2(0, -1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w)
		chkxer2("Dgebd2", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Dgebd2(2, 1, a.Off(0, 0).UpdateRows(1), d, e, tq, tp, w)
		chkxer2("Dgebd2", err)
		nt = nt + 3

		//        Dorgbr
		*srnamt = "Dorgbr"
		*errt = fmt.Errorf("!wantq && vect != 'P': vect='/'")
		err = golapack.Dorgbr('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dorgbr('Q', -1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', n=-1, k=0, m=0")
		err = golapack.Dorgbr('Q', 0, -1, 0, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', n=1, k=0, m=0")
		err = golapack.Dorgbr('Q', 0, 1, 0, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='Q', n=0, k=1, m=1")
		err = golapack.Dorgbr('Q', 1, 0, 1, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='P', n=0, k=0, m=1")
		err = golapack.Dorgbr('P', 1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='P', n=1, k=1, m=0")
		err = golapack.Dorgbr('P', 0, 1, 1, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("k < 0: k=-1")
		err = golapack.Dorgbr('Q', 0, 0, -1, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Dorgbr('Q', 2, 1, 1, a.Off(0, 0).UpdateRows(1), tq, w, 1)
		chkxer2("Dorgbr", err)
		*errt = fmt.Errorf("lwork < max(1, mn) && !lquery: lwork=1, mn=2, lquery=false")
		err = golapack.Dorgbr('Q', 2, 2, 1, a.Off(0, 0).UpdateRows(2), tq, w, 1)
		chkxer2("Dorgbr", err)
		nt = nt + 10

		//        Dormbr
		*srnamt = "Dormbr"
		*errt = fmt.Errorf("!applyq && vect != 'P': vect='/'")
		err = golapack.Dormbr('/', Left, Trans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
		err = golapack.Dormbr('Q', '/', Trans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
		err = golapack.Dormbr('Q', Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dormbr('Q', Left, Trans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dormbr('Q', Left, Trans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("k < 0: k=-1")
		err = golapack.Dormbr('Q', Left, Trans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='Q', a.Rows=1, nq=2, k=0")
		err = golapack.Dormbr('Q', Left, Trans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='Q', a.Rows=1, nq=2, k=0")
		err = golapack.Dormbr('Q', Right, Trans, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='P', a.Rows=1, nq=2, k=2")
		err = golapack.Dormbr('P', Left, Trans, 2, 0, 2, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("(applyq && a.Rows < max(1, nq)) || (!applyq && a.Rows < max(1, min(nq, k))): vect='P', a.Rows=1, nq=2, k=2")
		err = golapack.Dormbr('P', Right, Trans, 0, 2, 2, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
		err = golapack.Dormbr('Q', Right, Trans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Dormbr('Q', Left, Trans, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormbr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Dormbr('Q', Right, Trans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), tq, u.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dormbr", err)
		nt = nt + 13

		//        Dbdsqr
		*srnamt = "Dbdsqr"
		*errt = fmt.Errorf("uplo != Upper && !lower: uplo=Unrecognized: /")
		_, err = golapack.Dbdsqr('/', 0, 0, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dbdsqr(Upper, -1, 0, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("ncvt < 0: ncvt=-1")
		_, err = golapack.Dbdsqr(Upper, 0, -1, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("nru < 0: nru=-1")
		_, err = golapack.Dbdsqr(Upper, 0, 0, -1, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("ncc < 0: ncc=-1")
		_, err = golapack.Dbdsqr(Upper, 0, 0, 0, -1, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("(ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)): vt.Rows=1, n=2, ncvt=1")
		_, err = golapack.Dbdsqr(Upper, 2, 1, 0, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("u.Rows < max(1, nru): u.Rows=1, nru=2")
		_, err = golapack.Dbdsqr(Upper, 0, 0, 2, 0, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		*errt = fmt.Errorf("(ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)): c.Rows=1, n=2, ncc=1")
		_, err = golapack.Dbdsqr(Upper, 2, 0, 0, 1, d, e, v.Off(0, 0).UpdateRows(1), u.Off(0, 0).UpdateRows(1), a.Off(0, 0).UpdateRows(1), w)
		chkxer2("Dbdsqr", err)
		nt = nt + 8

		//        Dbdsdc
		*srnamt = "Dbdsdc"
		*errt = fmt.Errorf("iuplo == 0: uplo=Unrecognized: /")
		err = golapack.Dbdsdc('/', 'N', 0, d, e, u.Off(0, 0).UpdateRows(1), v.Off(0, 0).UpdateRows(1), q.OffIdx(0).Vector(), &iq, w, &iw)
		chkxer2("Dbdsdc", err)
		*errt = fmt.Errorf("icompq < 0: compq='/'")
		err = golapack.Dbdsdc(Upper, '/', 0, d, e, u.Off(0, 0).UpdateRows(1), v.Off(0, 0).UpdateRows(1), q.OffIdx(0).Vector(), &iq, w, &iw)
		chkxer2("Dbdsdc", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dbdsdc(Upper, 'N', -1, d, e, u.Off(0, 0).UpdateRows(1), v.Off(0, 0).UpdateRows(1), q.OffIdx(0).Vector(), &iq, w, &iw)
		chkxer2("Dbdsdc", err)
		*errt = fmt.Errorf("(u.Rows < 1) || ((icompq == 2) && (u.Rows < n)): compq='I', u.Rows=1, n=2")
		err = golapack.Dbdsdc(Upper, 'I', 2, d, e, u.Off(0, 0).UpdateRows(1), v.Off(0, 0).UpdateRows(1), q.OffIdx(0).Vector(), &iq, w, &iw)
		chkxer2("Dbdsdc", err)
		*errt = fmt.Errorf("(vt.Rows < 1) || ((icompq == 2) && (vt.Rows < n)): compq='I', vt.Rows=1, n=2")
		err = golapack.Dbdsdc(Upper, 'I', 2, d, e, u.Off(0, 0).UpdateRows(2), v.Off(0, 0).UpdateRows(1), q.OffIdx(0).Vector(), &iq, w, &iw)
		chkxer2("Dbdsdc", err)
		nt = nt + 5

		//        Dbdsvdx
		*srnamt = "Dbdsvdx"
		*errt = fmt.Errorf("uplo != Upper && !lower: uplo=Unrecognized: X")
		_, err = golapack.Dbdsvdx('X', 'N', 'A', 1, d, e, zero, one, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("!(wantz || jobz == 'N'): jobz='X'")
		_, err = golapack.Dbdsvdx(Upper, 'X', 'A', 1, d, e, zero, one, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("!(allsv || valsv || indsv): _range='X'")
		_, err = golapack.Dbdsvdx(Upper, 'V', 'X', 1, d, e, zero, one, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dbdsvdx(Upper, 'V', 'A', -1, d, e, zero, one, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("vl < zero: _range='V', vl=-1")
		_, err = golapack.Dbdsvdx(Upper, 'V', 'V', 2, d, e, -one, zero, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("vu <= vl: _range='V', vl=1, vu=0")
		_, err = golapack.Dbdsvdx(Upper, 'V', 'V', 2, d, e, one, zero, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("il < 1 || il > max(1, n): _range='I', n=2, il=0")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'I', 2, d, e, zero, zero, 0, 2, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("il < 1 || il > max(1, n): _range='I', n=4, il=5")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'I', 4, d, e, zero, zero, 5, 2, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("iu < min(n, il) || iu > n: _range='I', n=4, il=3, iu=2")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'I', 4, d, e, zero, zero, 3, 2, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("iu < min(n, il) || iu > n: _range='I', n=4, il=3, iu=5")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'I', 4, d, e, zero, zero, 3, 5, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n*2): jobz='V', z.Rows=1, n=4")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'A', 4, d, e, zero, zero, 0, 0, ns, s, q.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dbdsvdx", err)
		*errt = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n*2): jobz='V', z.Rows=2, n=4")
		_, err = golapack.Dbdsvdx(Lower, 'V', 'A', 4, d, e, zero, zero, 0, 0, ns, s, q.Off(0, 0).UpdateRows(2), w, &iw)
		chkxer2("Dbdsvdx", err)
		nt = nt + 12
	}

	//     Print a summary line.
	// if *ok {
	// 	fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	// } else {
	// 	fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	// }

	if !(*ok) {
		t.Fail()
	}

	return
}
