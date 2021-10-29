package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrorhrcol tests the error exits for DorhrCol that does
// Householder reconstruction from the ouput of tall-skinny
// factorization DLATSQR.
func derrorhrCol(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	t := mf(2, 2, opts)
	d := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		d.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for Householder reconstruction
	//
	//     DorhrCol
	*srnamt = "DorhrCol"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.DorhrCol(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.DorhrCol(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.DorhrCol(1, 2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("nb < 1: nb=-1")
	err = golapack.DorhrCol(0, 0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("nb < 1: nb=0")
	err = golapack.DorhrCol(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=-1, m=0")
	err = golapack.DorhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(-1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.DorhrCol(2, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=-1, nb=1, n=0")
	err = golapack.DorhrCol(0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(-1), d)
	chkxer2("DorhrCol", err)
	*errt = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=1, nb=2, n=3")
	err = golapack.DorhrCol(4, 3, 2, a.Off(0, 0).UpdateRows(4), t.Off(0, 0).UpdateRows(1), d)
	chkxer2("DorhrCol", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
