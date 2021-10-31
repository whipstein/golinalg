package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrqrtp tests the error exits for the REAL routines
// that use the QRT decomposition of a triangular-pentagonal matrix.
func derrqrtp(path string, _t *testing.T) {
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

	//     Error exits for TPQRT factorization
	//
	//     Dtpqrt
	*srnamt = "Dtpqrt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtpqrt(-1, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtpqrt(1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=-1, m=0, n=1")
	err = golapack.Dtpqrt(0, 1, -1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=1, m=0, n=1")
	err = golapack.Dtpqrt(0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > n && n > 0): nb=0, n=1")
	err = golapack.Dtpqrt(0, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > n && n > 0): nb=2, n=1")
	err = golapack.Dtpqrt(0, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	err = golapack.Dtpqrt(1, 2, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtpqrt(2, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Dtpqrt(2, 2, 1, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpqrt", err)

	//     Dtpqrt2
	*srnamt = "Dtpqrt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtpqrt2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtpqrt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtpqrt2(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtpqrt2", err)
	*errt = fmt.Errorf("l < 0 || l > min(m, n): l=-1, m=0, n=0")
	err = golapack.Dtpqrt2(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtpqrt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	err = golapack.Dtpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2))
	chkxer2("Dtpqrt2", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(2))
	chkxer2("Dtpqrt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Dtpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Dtpqrt2", err)

	//     Dtpmqrt
	*srnamt = "Dtpmqrt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dtpmqrt('/', NoTrans, 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dtpmqrt(Left, '/', 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dtpmqrt(Left, NoTrans, -1, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dtpmqrt(Left, NoTrans, 0, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Dtpmqrt(Left, NoTrans, 0, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("l < 0 || l > k: l=-1, k=0")
	err = golapack.Dtpmqrt(Left, NoTrans, 0, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > k && k > 0): nb=0, k=0")
	err = golapack.Dtpmqrt(Left, NoTrans, 0, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("v.Rows < ldvq: v.Rows=1, ldvq=2")
	err = golapack.Dtpmqrt(Right, NoTrans, 1, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("v.Rows < ldvq: v.Rows=1, ldvq=2")
	err = golapack.Dtpmqrt(Left, NoTrans, 2, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Dtpmqrt(Right, NoTrans, 1, 1, 2, 1, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("a.Rows < ldaq: a.Rows=1, ldaq=2")
	err = golapack.Dtpmqrt(Left, NoTrans, 1, 2, 2, 1, 2, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Dtpmqrt(Left, NoTrans, 2, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dtpmqrt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
