package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerlqt tests the error exits for the COMPLEX routines
// that use the LQT decomposition of a general matrix.
func zerrlqt(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	w := cvf(2)
	a := cmf(2, 2, opts)
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
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for LQT factorization
	//
	//     Zgelqt
	*srnamt = "Zgelqt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgelqt(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgelqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgelqt(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgelqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > min(m, n) && min(m, n) > 0): mb=0, m=0, n=0")
	err = golapack.Zgelqt(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgelqt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgelqt(2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgelqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Zgelqt(2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgelqt", err)

	//     Zgelqt3
	*srnamt = "Zgelqt3"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgelqt3(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgelqt3", err)
	*errt = fmt.Errorf("n < m: m=0, n=-1")
	err = golapack.Zgelqt3(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgelqt3", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgelqt3(2, 2, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgelqt3", err)
	*errt = fmt.Errorf("t.Rows < max(1, m): t.Rows=1, m=2")
	err = golapack.Zgelqt3(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgelqt3", err)

	//     Zgemlqt
	*srnamt = "Zgemlqt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Zgemlqt('/', NoTrans, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Zgemlqt(Left, '/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgemlqt(Left, NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgemlqt(Left, NoTrans, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Zgemlqt(Left, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Zgemlqt(Right, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > k && k > 0): mb=0, k=0")
	err = golapack.Zgemlqt(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("v.Rows < max(1, k): v.Rows=1, k=2")
	err = golapack.Zgemlqt(Right, NoTrans, 2, 2, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("v.Rows < max(1, k): v.Rows=1, k=2")
	err = golapack.Zgemlqt(Left, NoTrans, 2, 2, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Zgemlqt(Right, NoTrans, 1, 1, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zgemlqt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemlqt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
