package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrqrt tests the error exits for the DOUBLE PRECISION routines
// that use the QRT decomposition of a general matrix.
func derrqrt(path string, _t *testing.T) {
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

	//     Error exits for QRT factorization
	//
	//     Dgeqrt
	*srnamt = "Dgeqrt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqrt(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgeqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqrt(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgeqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > min(m, n) && min(m, n) > 0): m=0, n=0, nb=0")
	err = golapack.Dgeqrt(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgeqrt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqrt(2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgeqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Dgeqrt(2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgeqrt", err)

	//     Dgeqrt2
	*srnamt = "Dgeqrt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqrt2(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqrt2(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqrt2(2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Dgeqrt2(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt2", err)

	//     Dgeqrt3
	*srnamt = "Dgeqrt3"
	*errt = fmt.Errorf("m < n: m=-1, n=0")
	err = golapack.Dgeqrt3(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt3", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqrt3(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt3", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqrt3(2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt3", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Dgeqrt3(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dgeqrt3", err)

	//     Dgemqrt
	*srnamt = "Dgemqrt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dgemqrt('/', NoTrans, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dgemqrt(Left, '/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgemqrt(Left, NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgemqrt(Left, NoTrans, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("k < 0 || k > q: k=-1, q=0")
	err = golapack.Dgemqrt(Left, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("k < 0 || k > q: k=-1, q=0")
	err = golapack.Dgemqrt(Right, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > k && k > 0): k=0, nb=0")
	err = golapack.Dgemqrt(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("v.Rows < max(1, q): v.Rows=1, q=2")
	err = golapack.Dgemqrt(Right, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("v.Rows < max(1, q): v.Rows=1, q=2")
	err = golapack.Dgemqrt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Dgemqrt(Right, NoTrans, 2, 2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dgemqrt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dgemqrt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
