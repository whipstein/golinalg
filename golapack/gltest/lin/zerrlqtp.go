package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrlqtp tests the error exits for the complex routines
// that use the LQT decomposition of a triangular-pentagonal matrix.
func zerrlqtp(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	w := cvf(2)
	a := cmf(2, 2, opts)
	b := cmf(2, 2, opts)
	c := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2
	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			c.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			t.Set(i-1, j-1, 1./complex(float64(i+j), 0))
		}
		w.Set(j-1, 0.0)
	}
	(*ok) = true

	//     Error exits for TPLQT factorization
	//
	//     Ztplqt
	*srnamt = "Ztplqt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztplqt(-1, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztplqt(1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=-1, m=0, n=1")
	err = golapack.Ztplqt(0, 1, -1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=1, m=0, n=1")
	err = golapack.Ztplqt(0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > m && m > 0): mb=0, m=0")
	err = golapack.Ztplqt(0, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > m && m > 0): mb=2, m=1")
	err = golapack.Ztplqt(1, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Ztplqt(2, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Ztplqt(2, 1, 0, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Ztplqt(2, 2, 1, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztplqt", err)

	//     Ztplqt2
	*srnamt = "Ztplqt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztplqt2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztplqt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztplqt2(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztplqt2", err)
	*errt = fmt.Errorf("l < 0 || l > min(m, n): l=-1, m=0, n=0")
	err = golapack.Ztplqt2(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztplqt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Ztplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2))
	chkxer2("Ztplqt2", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Ztplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(2))
	chkxer2("Ztplqt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, m): t.Rows=1, m=2")
	err = golapack.Ztplqt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztplqt2", err)

	//     Ztpmlqt
	*srnamt = "Ztpmlqt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Ztpmlqt('/', NoTrans, 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Ztpmlqt(Left, '/', 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztpmlqt(Left, NoTrans, -1, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztpmlqt(Left, NoTrans, 0, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Ztpmlqt(Left, NoTrans, 0, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("l < 0 || l > k: l=-1, k=0")
	err = golapack.Ztpmlqt(Left, NoTrans, 0, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > k && k > 0): mb=0, k=0")
	err = golapack.Ztpmlqt(Left, NoTrans, 0, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("v.Rows < k: v.Rows=1, k=2")
	err = golapack.Ztpmlqt(Right, NoTrans, 2, 2, 2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)
	// *errt = fmt.Errorf("t.Rows < mb: t.Rows=0, mb=1")
	// err = golapack.Ztpmlqt(Right, NoTrans, 1, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(0), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	// chkxer2("Ztpmlqt", err)
	// *errt = fmt.Errorf("a.Rows < ldaq: a.Rows=0, ldaq=1")
	// err = golapack.Ztpmlqt(Left, NoTrans, 1, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(0), c.Off(0, 0).UpdateRows(1), w)
	// chkxer2("Ztpmlqt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Ztpmlqt(Left, NoTrans, 2, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmlqt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
