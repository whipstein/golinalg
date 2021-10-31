package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrhs tests the error exits for Zgebak, CGEBAL, CGEHRD, Zunghr,
// Zunmhr, Zhseqr, CHSEIN, and Ztrevc.
func zerrhs(path string, t *testing.T) (nt int) {
	var i, info, j, lw, nmax int
	var err error

	nmax = 3
	lw = nmax * nmax
	sel := make([]bool, 3)
	tau := cvf(3)
	w := cvf(lw)
	x := cvf(3)
	rw := vf(3)
	s := vf(3)
	ifaill := make([]int, 3)
	ifailr := make([]int, 3)
	a := cmf(3, 3, opts)
	c := cmf(3, 3, opts)
	vl := cmf(3, 3, opts)
	vr := cmf(3, 3, opts)

	errt := &gltest.Common.Infoc.Errt
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
		sel[j-1] = true
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the nonsymmetric eigenvalue routines.
	if c2 == "hs" {
		//        Zgebal
		*srnamt = "Zgebal"
		*errt = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='/'")
		_, _, err = golapack.Zgebal('/', 0, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Zgebal", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zgebal('N', -1, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Zgebal", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Zgebal('N', 2, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Zgebal", err)
		nt = nt + 3

		//        Zgebak
		*srnamt = "Zgebak"
		*errt = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='/'")
		err = golapack.Zgebak('/', Right, 0, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		err = golapack.Zgebak('N', '/', 0, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgebak('N', Right, -1, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Zgebak('N', Right, 0, 0, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Zgebak('N', Right, 0, 2, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=2, ihi=1, n=2")
		err = golapack.Zgebak('N', Right, 2, 2, 1, s, 0, a.Off(0, 0).UpdateRows(2))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=1, n=0")
		err = golapack.Zgebak('N', Right, 0, 1, 1, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Zgebak('N', Right, 0, 1, 0, s, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		*errt = fmt.Errorf("v.Rows < max(1, n): v.Rows=1, n=2")
		err = golapack.Zgebak('N', Right, 2, 1, 2, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zgebak", err)
		nt = nt + 9

		//        Zgehrd
		*srnamt = "Zgehrd"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgehrd(-1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Zgehrd(0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Zgehrd(0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=0, n=1")
		err = golapack.Zgehrd(1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=1, n=0")
		err = golapack.Zgehrd(0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zgehrd(2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 2)
		chkxer2("Zgehrd", err)
		*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
		err = golapack.Zgehrd(2, 1, 2, a.Off(0, 0).UpdateRows(2), tau, w, 1)
		chkxer2("Zgehrd", err)
		nt = nt + 7

		//        Zunghr
		*srnamt = "Zunghr"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zunghr(-1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Zunghr(0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Zunghr(0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=0, n=1")
		err = golapack.Zunghr(1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=1, n=0")
		err = golapack.Zunghr(0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zunghr(2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		*errt = fmt.Errorf("lwork < max(1, nh) && !lquery: lwork=1, nh=2, lquery=false")
		err = golapack.Zunghr(3, 1, 3, a.Off(0, 0).UpdateRows(3), tau, w.Off(0, 1), 1)
		chkxer2("Zunghr", err)
		nt = nt + 7

		//        Zunmhr
		*srnamt = "Zunmhr"
		*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
		err = golapack.Zunmhr('/', NoTrans, 0, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zunmhr(Left, '/', 0, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Zunmhr(Left, NoTrans, -1, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zunmhr(Left, NoTrans, 0, -1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=0, ihi=0, nq=0")
		err = golapack.Zunmhr(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, ihi=0, nq=0")
		err = golapack.Zunmhr(Left, NoTrans, 0, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, ihi=1, nq=1")
		err = golapack.Zunmhr(Left, NoTrans, 1, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 2), 2)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, ihi=1, nq=1")
		err = golapack.Zunmhr(Right, NoTrans, 2, 1, 2, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w.Off(0, 2), 2)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ilo=1, ihi=0, nq=1")
		err = golapack.Zunmhr(Left, NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ilo=1, ihi=1, nq=0")
		err = golapack.Zunmhr(Left, NoTrans, 0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ilo=1, ihi=1, nq=0")
		err = golapack.Zunmhr(Right, NoTrans, 1, 0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
		err = golapack.Zunmhr(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
		err = golapack.Zunmhr(Right, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
		err = golapack.Zunmhr(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Zunmhr(Left, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Zunmhr(Right, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
		chkxer2("Zunmhr", err)
		nt = nt + 16

		//        Zhseqr
		*srnamt = "Zhseqr"
		// *errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		*errt = fmt.Errorf("job != 'E' && !wantt: job='/'")
		_, err = golapack.Zhseqr('/', 'N', 0, 1, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("compz != 'N' && !wantz: compz='/'")
		_, err = golapack.Zhseqr('E', '/', 0, 1, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhseqr('E', 'N', -1, 1, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=0, ilo=0")
		_, err = golapack.Zhseqr('E', 'N', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=0, ilo=2")
		_, err = golapack.Zhseqr('E', 'N', 0, 2, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=1, ilo=1, ihi=0")
		_, err = golapack.Zhseqr('E', 'N', 1, 1, 0, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=1, ilo=1, ihi=2")
		_, err = golapack.Zhseqr('E', 'N', 1, 1, 2, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("h.Rows < max(1, n): h.Rows=1, n=2")
		_, err = golapack.Zhseqr('E', 'N', 2, 1, 2, a.Off(0, 0).UpdateRows(1), x, c.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Zhseqr", err)
		*errt = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < max(1, n)): compz='V', z.Rows=1, n=2")
		_, err = golapack.Zhseqr('E', 'V', 2, 1, 2, a.Off(0, 0).UpdateRows(2), x, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zhseqr", err)
		nt = nt + 9

		//        Zhsein
		*srnamt = "Zhsein"
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		_, _, err = golapack.Zhsein('/', 'N', 'N', sel, 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("!fromqr && eigsrc != 'N': eigsrc='/'")
		_, _, err = golapack.Zhsein(Right, '/', 'N', sel, 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("!noinit && initv != 'U': initv='/'")
		_, _, err = golapack.Zhsein(Right, 'N', '/', sel, 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zhsein(Right, 'N', 'N', sel, -1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("h.Rows < max(1, n): h.Rows=1, n=2")
		_, _, err = golapack.Zhsein(Right, 'N', 'N', sel, 2, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(2), 4, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=Left, vl.Rows=1, n=2")
		_, _, err = golapack.Zhsein(Left, 'N', 'N', sel, 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=Right, vr.Rows=1, n=2")
		_, _, err = golapack.Zhsein(Right, 'N', 'N', sel, 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		*errt = fmt.Errorf("mm < m: mm=1, m=2")
		_, _, err = golapack.Zhsein(Right, 'N', 'N', sel, 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(2), 1, w, rw, &ifaill, &ifailr)
		chkxer2("Zhsein", err)
		nt = nt + 8

		//        Ztrevc
		*srnamt = "Ztrevc"
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		_, err = golapack.Ztrevc('/', 'A', sel, 0, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("!allv && !over && !somev: howmny='/'")
		_, err = golapack.Ztrevc(Left, '/', sel, 0, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztrevc(Left, 'A', sel, -1, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
		_, err = golapack.Ztrevc(Left, 'A', sel, 2, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(2), vr.Off(0, 0).UpdateRows(1), 4, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=Left, vl.Rows=1, n=2")
		_, err = golapack.Ztrevc(Left, 'A', sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=Right, vr.Rows=1, n=2")
		_, err = golapack.Ztrevc(Right, 'A', sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		*errt = fmt.Errorf("mm < m: mm=1, m=2")
		_, err = golapack.Ztrevc(Left, 'A', sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(2), vr.Off(0, 0).UpdateRows(1), 1, w, rw)
		chkxer("Ztrevc", info, lerr, ok, t)
		nt = nt + 7
	}

	//     Print a summary line.
	// if *ok {
	// 	fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	// } else {
	// 	fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	// }
	*infot = 0
	*srnamt = ""
	if !(*ok) {
		t.Fail()
	}
	return
}
