package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrlqtp tests the error exits for the REAL routines
// that use the LQT decomposition of a triangular-pentagonal matrix.
func derrlqtp(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	b := mf(2, 2, opts)
	c := mf(2, 2, opts)
	t := mf(2, 2, opts)
	w := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			c.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.0)
	}
	(*ok) = true

	//     Error exits for TPLQT factorization
	//
	//     Dtplqt
	*srnamt = "Dtplqt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtplqt(-1, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtplqt(1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=-1, m=0, n=1")
	err = golapack.Dtplqt(0, 1, -1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=1, m=0, n=1")
	err = golapack.Dtplqt(0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > m && m > 0): mb=0, m=0")
	err = golapack.Dtplqt(0, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > m && m > 0): mb=2, m=1")
	err = golapack.Dtplqt(1, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dtplqt(2, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtplqt(2, 1, 0, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Dtplqt(2, 2, 1, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtplqt", err)

	//     Dtplqt2
	*srnamt = "Dtplqt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtplqt2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtplqt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtplqt2(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtplqt2", err)
	*errt = fmt.Errorf("l < 0 || l > min(m, n): l=-1, m=0, n=0")
	err = golapack.Dtplqt2(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtplqt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dtplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2))
	chkxer2("Dtplqt2", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(2))
	chkxer2("Dtplqt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, m): t.Rows=1, m=2")
	err = golapack.Dtplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtplqt2", err)

	//     Dtpmlqt
	*srnamt = "Dtpmlqt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dtpmlqt('/', NoTrans, 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dtpmlqt(Left, '/', 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtpmlqt(Left, NoTrans, -1, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtpmlqt(Left, NoTrans, 0, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Dtpmlqt(Left, NoTrans, 0, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("l < 0 || l > k: l=-1, k=0")
	err = golapack.Dtpmlqt(Left, NoTrans, 0, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > k && k > 0): mb=0, k=0")
	err = golapack.Dtpmlqt(Left, NoTrans, 0, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("v.Rows < k: v.Rows=1, k=2")
	err = golapack.Dtpmlqt(Right, NoTrans, 2, 2, 2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Dtpmlqt(Right, NoTrans, 1, 1, 2, 1, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("a.Rows < ldaq: a.Rows=1, ldaq=2")
	err = golapack.Dtpmlqt(Left, NoTrans, 2, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtpmlqt(Left, NoTrans, 2, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmlqt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
