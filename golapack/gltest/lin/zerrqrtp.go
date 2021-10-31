package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrqrtp tests the error exits for the COMPLEX*16 routines
// that use the QRT decomposition of a triangular-pentagonal matrix.
func zerrqrtp(path string, _t *testing.T) {
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
		w.Set(j-1, complex(0, 0))
	}
	(*ok) = true

	//     Error exits for TPQRT factorization
	//
	//     Ztpqrt
	*srnamt = "Ztpqrt"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztpqrt(-1, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztpqrt(1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=-1, m=0, n=1")
	err = golapack.Ztpqrt(0, 1, -1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("l < 0 || (l > min(m, n) && min(m, n) >= 0): l=1, m=0, n=1")
	err = golapack.Ztpqrt(0, 1, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > n && n > 0): nb=0, n=1")
	err = golapack.Ztpqrt(0, 1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > n && n > 0): nb=2, n=1")
	err = golapack.Ztpqrt(0, 1, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	err = golapack.Ztpqrt(1, 2, 0, 2, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Ztpqrt(2, 1, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Ztpqrt(2, 2, 1, 2, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpqrt", err)
	//
	//     Ztpqrt2
	//
	*srnamt = "Ztpqrt2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztpqrt2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztpqrt2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztpqrt2(0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztpqrt2", err)
	*errt = fmt.Errorf("l < 0 || l > min(m, n): l=-1, m=0, n=0")
	err = golapack.Ztpqrt2(0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztpqrt2", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	err = golapack.Ztpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(2))
	chkxer2("Ztpqrt2", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = golapack.Ztpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(2))
	chkxer2("Ztpqrt2", err)
	*errt = fmt.Errorf("t.Rows < max(1, n): t.Rows=1, n=2")
	err = golapack.Ztpqrt2(2, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), t.Off(0, 0).UpdateRows(1))
	chkxer2("Ztpqrt2", err)
	//
	//     Ztpmqrt
	//
	*srnamt = "Ztpmqrt"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Ztpmqrt('/', NoTrans, 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Ztpmqrt(Left, '/', 0, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Ztpmqrt(Left, NoTrans, -1, 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Ztpmqrt(Left, NoTrans, 0, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("k < 0: k=-1")
	err = golapack.Ztpmqrt(Left, NoTrans, 0, 0, -1, 0, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("l < 0 || l > k: l=-1, k=0")
	err = golapack.Ztpmqrt(Left, NoTrans, 0, 0, 0, -1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("nb < 1 || (nb > k && k > 0): nb=0, k=0")
	err = golapack.Ztpmqrt(Left, NoTrans, 0, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("v.Rows < ldvq: v.Rows=1, ldvq=2")
	err = golapack.Ztpmqrt(Right, NoTrans, 1, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("v.Rows < ldvq: v.Rows=1, ldvq=2")
	err = golapack.Ztpmqrt(Left, NoTrans, 2, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	*errt = fmt.Errorf("t.Rows < nb: t.Rows=1, nb=2")
	err = golapack.Ztpmqrt(Right, NoTrans, 1, 1, 3, 1, 2, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(1), w)
	chkxer2("Ztpmqrt", err)
	// *errt = fmt.Errorf("a.Rows < ldaq: a.Rows=0, ldaq=1")
	// err = golapack.Ztpmqrt(Left, NoTrans, 1, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(0), c.Off(0, 0).UpdateRows(1), w)
	// chkxer2("Ztpmqrt", err)
	// *errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=0, m=1")
	// err = golapack.Ztpmqrt(Left, NoTrans, 1, 1, 1, 1, 1, a.Off(0, 0).UpdateRows(1), t.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), c.Off(0, 0).UpdateRows(0), w)
	// chkxer2("Ztpmqrt", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
