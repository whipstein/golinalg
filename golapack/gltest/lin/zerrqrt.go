package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrqrt tests the error exits for the COMPLEX*16 routines
// that use the QRT decomposition of a general matrix.
func zerrqrt(path string, _t *testing.T) {
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

	//     Error exits for QRT factorization
	//
	//     Zgeqrt
	*srnamt = "Zgeqrt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqrt(-1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgeqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqrt(0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgeqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > min(m, n) && min(m, n) > 0): nb=0, m=0, n=0")
	err = golapack.Zgeqrt(0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgeqrt", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqrt(2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgeqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Zgeqrt(2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgeqrt", err)

	//     Zgeqrt2
	*srnamt = "Zgeqrt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqrt2(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqrt2(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqrt2(2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Zgeqrt2(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt2", err)

	//     Zgeqrt3
	*srnamt = "Zgeqrt3"
	*errt = fmt.Errorf("m < n: m=-1")
	err = golapack.Zgeqrt3(-1, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt3", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqrt3(0, -1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt3", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqrt3(2, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt3", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Zgeqrt3(2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Zgeqrt3", err)

	//     Zgemqrt
	*srnamt = "Zgemqrt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Zgemqrt('/', NoTrans, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Zgemqrt(Left, '/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgemqrt(Left, NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgemqrt(Left, NoTrans, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("k < 0 || k > q: k=-1")
	err = golapack.Zgemqrt(Left, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("k < 0 || k > q: k=-1")
	err = golapack.Zgemqrt(Right, NoTrans, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > k && k > 0): nb=0, k=0")
	err = golapack.Zgemqrt(Left, NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("v.Rows < max(1, q): v.Rows=1, q=2")
	err = golapack.Zgemqrt(Right, NoTrans, 1, 2, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("v.Rows < max(1, q): v.Rows=1, q=2")
	err = golapack.Zgemqrt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Zgemqrt(Right, NoTrans, 2, 2, 2, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(2), w)
	chkxer2("Zgemqrt", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zgemqrt(Left, NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zgemqrt", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
