package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derlqt tests the error exits for the DOUBLE PRECISION routines
// that use the LQT decomposition of a general matrix.
func derrlqt(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
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
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for LQT factorization
	//
	//     Dgelqt
	*srnamt = "Dgelqt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgelqt(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgelqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgelqt(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgelqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > min(m, n) && min(m, n) > 0): m=0, n=0, mb=0")
	err = golapack.Dgelqt(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgelqt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgelqt(2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgelqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Dgelqt(2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgelqt", err)

	//     Dgelqt3
	*srnamt = "Dgelqt3"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgelqt3(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgelqt3", err)
	*errt = fmt.Errorf("n < m: m=0, n=-1")
	err = golapack.Dgelqt3(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgelqt3", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgelqt3(2, 2, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgelqt3", err)
	*errt = fmt.Errorf("t.Rows < max(1, m): t.Rows=1, m=2")
	err = golapack.Dgelqt3(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgelqt3", err)

	//     Dgemlqt
	*srnamt = "Dgemlqt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dgemlqt('/', NoTrans, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dgemlqt(Left, '/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgemlqt(Left, NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgemlqt(Left, NoTrans, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Dgemlqt(Left, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Dgemlqt(Right, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("mb < 1 || (mb > k && k > 0): k=0, mb=0")
	err = golapack.Dgemlqt(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("v.Rows < max(1, k): v.Rows=1, k=2")
	err = golapack.Dgemlqt(Right, NoTrans, 2, 2, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("v.Rows < max(1, k): v.Rows=1, k=2")
	err = golapack.Dgemlqt(Left, NoTrans, 2, 2, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("t.Rows < mb: t.Rows=1, mb=2")
	err = golapack.Dgemlqt(Right, NoTrans, 2, 2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dgemlqt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemlqt", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
