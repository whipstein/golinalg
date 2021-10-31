package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrls tests the error exits for the COMPLEX*16 least squares
// driver routines (ZGELS, CGELSS, CGELSY, CGELSD).
func zerrls(path string, t *testing.T) {
	var rcond float64
	var err error

	w := cvf(2)
	rw := vf(2)
	s := vf(2)
	ip := make([]int, 2)
	a := cmf(2, 2, opts)
	b := cmf(2, 2, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	a.Set(0, 0, (1.0 + 0.0*1i))
	a.Set(0, 1, (2.0 + 0.0*1i))
	a.Set(1, 1, (3.0 + 0.0*1i))
	a.Set(1, 0, (4.0 + 0.0*1i))
	(*ok) = true

	//     Test error exits for the least squares driver routines.
	if c2 == "ls" {
		//        ZGELS
		*srnamt = "Zgels"
		*errt = fmt.Errorf("!(trans == NoTrans || trans == ConjTrans): trans=Unrecognized: /")
		_, err = golapack.Zgels('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgels(NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgels(NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zgels(NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgels(NoTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), w, 2)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=1, m=2, n=0")
		_, err = golapack.Zgels(NoTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), w, 2)
		chkxer2("Zgels", err)
		*errt = fmt.Errorf("lwork < max(1, mn+max(mn, nrhs)) && !lquery: lwork=1, mn=1, nrhs=0, lquery=false")
		_, err = golapack.Zgels(NoTrans, 1, 1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Zgels", err)

		//        Zgelss
		*srnamt = "Zgelss"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Zgelss(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1, rw)
		chkxer2("Zgelss", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zgelss(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1, rw)
		chkxer2("Zgelss", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zgelss(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1, rw)
		chkxer2("Zgelss", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, err = golapack.Zgelss(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), s, rcond, w, 2, rw)
		chkxer2("Zgelss", err)
		*errt = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=1, maxmn=2")
		_, _, err = golapack.Zgelss(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), s, rcond, w, 2, rw)
		chkxer2("Zgelss", err)

		//        Zgelsy
		*srnamt = "Zgelsy"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgelsy(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10, rw)
		chkxer2("Zgelsy", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgelsy(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10, rw)
		chkxer2("Zgelsy", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zgelsy(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10, rw)
		chkxer2("Zgelsy", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgelsy(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), &ip, rcond, w, 10, rw)
		chkxer2("Zgelsy", err)
		*errt = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=1, m=2, n=0")
		_, err = golapack.Zgelsy(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10, rw)
		chkxer2("Zgelsy", err)
		*errt = fmt.Errorf("lwork < (mn+max(2*mn, n+1, mn+nrhs)) && !lquery: lwork=1, mn=0, nrhs=0, lquery=false")
		_, err = golapack.Zgelsy(0, 3, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(3), &ip, rcond, w, 1, rw)
		chkxer2("Zgelsy", err)

		//        Zgelsd
		*srnamt = "Zgelsd"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Zgelsd(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, rw, &ip)
		chkxer2("Zgelsd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zgelsd(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, rw, &ip)
		chkxer2("Zgelsd", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zgelsd(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, rw, &ip)
		chkxer2("Zgelsd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, err = golapack.Zgelsd(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), s, rcond, w, 10, rw, &ip)
		chkxer2("Zgelsd", err)
		*errt = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=1, maxmn=2")
		_, _, err = golapack.Zgelsd(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, rw, &ip)
		chkxer2("Zgelsd", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=6, lquery=false")
		_, _, err = golapack.Zgelsd(2, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), s, rcond, w, 1, rw, &ip)
		chkxer2("Zgelsd", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
