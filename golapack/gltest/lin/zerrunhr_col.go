package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrunhrCol tests the error exits for ZunhrCol that does
// Householder reconstruction from the ouput of tall-skinny
// factorization ZLATSQR.
func zerrunhrCol(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	d := cvf(2)
	a := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))
			t.SetRe(i-1, j-1, 1./float64(i+j))
		}
		d.Set(j-1, (0. + 0.*1i))
	}
	(*ok) = true

	//     Error exits for Householder reconstruction
	//
	//     ZunhrCol
	*srnamt = "ZunhrCol"

	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.ZunhrCol(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)

	*errt = fmt.Errorf("n < 0 || n > m: m=0, n=-1")
	err = golapack.ZunhrCol(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=1, n=2")
	err = golapack.ZunhrCol(1, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)

	*errt = fmt.Errorf("nb < 1: nb=-1")
	err = golapack.ZunhrCol(0, 0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)
	*errt = fmt.Errorf("nb < 1: nb=0")
	err = golapack.ZunhrCol(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)

	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=-1, m=0")
	err = golapack.ZunhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(-1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)
	// *errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=0, m=0")
	// err = golapack.ZunhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(0), t.Off(0, 0).UpdateRows(1), d)
	// chkxer2("ZunhrCol", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.ZunhrCol(2, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)

	*errt = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=-1, n=0, nb=1")
	err = golapack.ZunhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(-1), d)
	chkxer2("ZunhrCol", err)
	// *errt = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=0, n=0, nb=1")
	// err = golapack.ZunhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(0), d)
	// chkxer2("ZunhrCol", err)
	*errt = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=1, n=3, nb=2")
	err = golapack.ZunhrCol(4, 3, 2, a.Off(0, 0).UpdateRows(4), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("ZunhrCol", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
