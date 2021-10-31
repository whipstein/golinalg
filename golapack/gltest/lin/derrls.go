package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrls tests the error exits for the DOUBLE PRECISION least squares
// driver routines (DGELS, SGELSS, SGELSY, SGELSD).
func derrls(path string, t *testing.T) {
	var rcond float64
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt
	ip := make([]int, 2)

	a := mf(2, 2, opts)
	b := mf(2, 2, opts)
	s := vf(2)
	w := vf(2)

	c2 := path[1:3]
	a.Set(0, 0, 1.0)
	a.Set(0, 1, 2.0)
	a.Set(1, 1, 3.0)
	a.Set(1, 0, 4.0)
	(*ok) = true

	if c2 == "ls" {
		//        Test error exits for the least squares driver routines.
		//
		//        DGELS
		*srnamt = "Dgels"
		*errt = fmt.Errorf("!(trans == NoTrans || trans == Trans): trans=Unrecognized: /")
		_, err = golapack.Dgels('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgels(NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgels(NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dgels(NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgels(NoTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), w, 2)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=1, m=2, n=0")
		_, err = golapack.Dgels(NoTrans, 2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), w, 2)
		chkxer2("Dgels", err)
		*errt = fmt.Errorf("lwork < max(1, mn+max(mn, nrhs)) && !lquery: lwork=1, m=1, n=1, nrhs=0, lquery=false")
		_, err = golapack.Dgels(NoTrans, 1, 1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgels", err)

		//        Dgelss
		*srnamt = "Dgelss"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Dgelss(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1)
		chkxer2("Dgelss", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgelss(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1)
		chkxer2("Dgelss", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dgelss(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 1)
		chkxer2("Dgelss", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, err = golapack.Dgelss(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), s, rcond, w, 2)
		chkxer2("Dgelss", err)
		*errt = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=1, m=2, n=0")
		_, _, err = golapack.Dgelss(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), s, rcond, w, 2)
		chkxer2("Dgelss", err)

		//        Dgelsy
		*srnamt = "Dgelsy"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgelsy(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10)
		chkxer2("Dgelsy", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgelsy(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10)
		chkxer2("Dgelsy", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dgelsy(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10)
		chkxer2("Dgelsy", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgelsy(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), &ip, rcond, w, 10)
		chkxer2("Dgelsy", err)
		*errt = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=1, m=2, n=0")
		_, err = golapack.Dgelsy(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), &ip, rcond, w, 10)
		chkxer2("Dgelsy", err)
		*errt = fmt.Errorf("lwork < lwkmin && !lquery: lwork=1, lwkmin=6, lquery=false")
		_, err = golapack.Dgelsy(2, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), &ip, rcond, w, 1)
		chkxer2("Dgelsy", err)

		//        Dgelsd
		*srnamt = "Dgelsd"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Dgelsd(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, &ip)
		chkxer2("Dgelsd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgelsd(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, &ip)
		chkxer2("Dgelsd", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dgelsd(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, &ip)
		chkxer2("Dgelsd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, err = golapack.Dgelsd(2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), s, rcond, w, 10, &ip)
		chkxer2("Dgelsd", err)
		*errt = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=1, m=2, n=0")
		_, _, err = golapack.Dgelsd(2, 0, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), s, rcond, w, 10, &ip)
		chkxer2("Dgelsd", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=802, lquery=false")
		_, _, err = golapack.Dgelsd(2, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), s, rcond, w, 1, &ip)
		chkxer2("Dgelsd", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
