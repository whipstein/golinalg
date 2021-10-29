package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrhs tests the error exits for Dgebak, SGEBAL, SGEHRD, Dorghr,
// Dormhr, Dhseqr, SHSEIN, and Dtrevc.
func derrhs(path string, t *testing.T) {
	var i, j, lw, nmax, nt int
	var err error
	sel := make([]bool, 3)
	ifaill := make([]int, 3)
	ifailr := make([]int, 3)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 3
	lw = (nmax+2)*(nmax+2) + nmax
	s := vf(nmax)
	tau := vf(nmax)
	w := vf(lw)
	wi := vf(nmax)
	wr := vf(nmax)
	a := mf(nmax, nmax, opts)
	c := mf(nmax, nmax, opts)
	vl := mf(nmax, nmax, opts)
	vr := mf(nmax, nmax, opts)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
		}
		wi.Set(j-1, float64(j))
		sel[j-1] = true
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the nonsymmetric eigenvalue routines.
	if c2 == "hs" {
		//        Dgebal
		*srnamt = "Dgebal"
		*errt = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='/'")
		_, _, err = golapack.Dgebal('/', 0, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Dgebal", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgebal('N', -1, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Dgebal", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Dgebal('N', 2, a.Off(0, 0).UpdateRows(1), s)
		chkxer2("Dgebal", err)
		nt = nt + 3

		//        Dgebak
		*srnamt = "Dgebak"
		*errt = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='/'")
		err = golapack.Dgebak('/', Right, 0, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		err = golapack.Dgebak('N', '/', 0, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgebak('N', Right, -1, 1, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Dgebak('N', Right, 0, 0, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Dgebak('N', Right, 0, 2, 0, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=2, ihi=1, n=2")
		err = golapack.Dgebak('N', Right, 2, 2, 1, s, 0, a.Off(0, 0).UpdateRows(2))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=1, n=0")
		err = golapack.Dgebak('N', Right, 0, 1, 1, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dgebak('N', Right, 0, 1, 0, s, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		*errt = fmt.Errorf("v.Rows < max(1, n): v.Rows=1, n=2")
		err = golapack.Dgebak('N', Right, 2, 1, 2, s, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dgebak", err)
		nt = nt + 9

		//        Dgehrd
		*srnamt = "Dgehrd"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgehrd(-1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Dgehrd(0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Dgehrd(0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=0, n=1")
		err = golapack.Dgehrd(1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=1, ihi=1, n=0")
		err = golapack.Dgehrd(0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dgehrd(2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 2)
		chkxer2("Dgehrd", err)
		*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
		err = golapack.Dgehrd(2, 1, 2, a.Off(0, 0).UpdateRows(2), tau, w, 1)
		chkxer2("Dgehrd", err)
		nt = nt + 7

		//        Dorghr
		*srnamt = "Dorghr"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dorghr(-1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=0, n=0")
		err = golapack.Dorghr(0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=2, n=0")
		err = golapack.Dorghr(0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ihi=0, ilo=1, n=1")
		err = golapack.Dorghr(1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ihi=1, ilo=1, n=0")
		err = golapack.Dorghr(0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dorghr(2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dorghr", err)
		*errt = fmt.Errorf("lwork < max(1, nh) && !lquery: lwork=1, nh=2, lquery=false")
		err = golapack.Dorghr(3, 1, 3, a.Off(0, 0).UpdateRows(3), tau, w, 1)
		chkxer2("Dorghr", err)
		nt = nt + 7

		//        Dormhr
		*srnamt = "Dormhr"
		*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
		err = golapack.Dormhr('/', NoTrans, 0, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans: trans=Unrecognized: /")
		err = golapack.Dormhr(Left, '/', 0, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dormhr(Left, NoTrans, -1, 0, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dormhr(Left, NoTrans, 0, -1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=0, nq=0")
		err = golapack.Dormhr(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, nq=0")
		err = golapack.Dormhr(Left, NoTrans, 0, 0, 2, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, nq=1")
		err = golapack.Dormhr(Left, NoTrans, 1, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 2)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, nq): ilo=2, nq=1")
		err = golapack.Dormhr(Right, NoTrans, 2, 1, 2, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w, 2)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ihi=0, ilo=1, nq=1")
		err = golapack.Dormhr(Left, NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ihi=1, ilo=1, nq=0")
		err = golapack.Dormhr(Left, NoTrans, 0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("ihi < min(ilo, nq) || ihi > nq: ihi=1, ilo=1, nq=0")
		err = golapack.Dormhr(Right, NoTrans, 1, 0, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
		err = golapack.Dormhr(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
		err = golapack.Dormhr(Right, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
		err = golapack.Dormhr(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Dormhr(Left, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dormhr", err)
		*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
		err = golapack.Dormhr(Right, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), tau, c.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dormhr", err)
		nt = nt + 16

		//        Dhseqr
		*srnamt = "Dhseqr"
		// *errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		*errt = fmt.Errorf("job != 'E' && !wantt: job='/'")
		_, err = golapack.Dhseqr('/', 'N', 0, 1, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("compz != 'N' && !wantz: compz='/'")
		_, err = golapack.Dhseqr('E', '/', 0, 1, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dhseqr('E', 'N', -1, 1, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=0, ilo=0, ihi=0")
		_, err = golapack.Dhseqr('E', 'N', 0, 0, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=0, ilo=2, ihi=0")
		_, err = golapack.Dhseqr('E', 'N', 0, 2, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=1, ilo=1, ihi=0")
		_, err = golapack.Dhseqr('E', 'N', 1, 1, 0, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=1, ilo=1, ihi=2")
		_, err = golapack.Dhseqr('E', 'N', 1, 1, 2, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("h.Rows < max(1, n): h.Rows=1, n=2")
		_, err = golapack.Dhseqr('E', 'N', 2, 1, 2, a.Off(0, 0).UpdateRows(1), wr, wi, c.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dhseqr", err)
		*errt = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < max(1, n)): compz='V', z.Rows=1, n=2")
		_, err = golapack.Dhseqr('E', 'V', 2, 1, 2, a.Off(0, 0).UpdateRows(2), wr, wi, c.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dhseqr", err)
		nt = nt + 9

		//        Dhsein
		*srnamt = "Dhsein"
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		_, _, err = golapack.Dhsein('/', 'N', 'N', &sel, 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("!fromqr && eigsrc != 'N': eigsrc='/'")
		_, _, err = golapack.Dhsein(Right, '/', 'N', &sel, 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("!noinit && initv != 'U': initv='/'")
		_, _, err = golapack.Dhsein(Right, 'N', '/', &sel, 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dhsein(Right, 'N', 'N', &sel, -1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("h.Rows < max(1, n): h.Rows=1, n=2")
		_, _, err = golapack.Dhsein(Right, 'N', 'N', &sel, 2, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(2), 4, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=Left, vl.Rows=1, n=2")
		_, _, err = golapack.Dhsein(Left, 'N', 'N', &sel, 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=Right, vr.Rows=1, n=2")
		_, _, err = golapack.Dhsein(Right, 'N', 'N', &sel, 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		*errt = fmt.Errorf("mm < m: mm=1, m=2")
		_, _, err = golapack.Dhsein(Right, 'N', 'N', &sel, 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(2), 1, w, &ifaill, &ifailr)
		chkxer2("Dhsein", err)
		nt = nt + 8

		//        Dtrevc
		*srnamt = "Dtrevc"
		*errt = fmt.Errorf("!rightv && !leftv: side=Unrecognized: /")
		_, err = golapack.Dtrevc('/', 'A', &sel, 0, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("!allv && !over && !somev: howmny='/'")
		_, err = golapack.Dtrevc(Left, '/', &sel, 0, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtrevc(Left, 'A', &sel, -1, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 0, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
		_, err = golapack.Dtrevc(Left, 'A', &sel, 2, a.Off(0, 0).UpdateRows(1), vl.Off(0, 0).UpdateRows(2), vr.Off(0, 0).UpdateRows(1), 4, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=Left, vl.Rows=1, n=2")
		_, err = golapack.Dtrevc(Left, 'A', &sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=Right, vr.Rows=1, n=2")
		_, err = golapack.Dtrevc(Right, 'A', &sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), 4, w)
		chkxer2("Dtrevc", err)
		*errt = fmt.Errorf("mm < m: mm=1, m=2")
		_, err = golapack.Dtrevc(Left, 'A', &sel, 2, a.Off(0, 0).UpdateRows(2), vl.Off(0, 0).UpdateRows(2), vr.Off(0, 0).UpdateRows(1), 1, w)
		chkxer2("Dtrevc", err)
		nt = nt + 7
	}

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
