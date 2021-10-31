package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrtsqr tests the error exits for the ZOUBLE PRECISION routines
// that use the TSQR decomposition of a general matrix.
func zerrtsqr(path string, _t *testing.T) {
	var i, j, nmax int
	var err error

	tau := cvf(4)
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
			a.SetRe(i-1, j-1, 1./float64(i+j))
			c.SetRe(i-1, j-1, 1./float64(i+j))
			t.SetRe(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for TS factorization
	//
	//     Zgeqr
	*srnamt = "Zgeqr"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqr(-1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgeqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqr(0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgeqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqr(2, 1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgeqr", err)
	*errt = fmt.Errorf("tsize < max(1, nb*n*nblcks+5) && (!lquery) && (!lminws): tsize=1, nb=1, n=2, nblcks=1, lquery=false, lminws=false")
	err = golapack.Zgeqr(3, 2, a.Off(0, 0).UpdateRows(3), tau, 1, w, 1)
	chkxer2("Zgeqr", err)
	*errt = fmt.Errorf("(lwork < max(1, n*nb)) && (!lquery) && (!lminws): lwork=0, n=2, nb=1, lquery=false, lminws=false")
	err = golapack.Zgeqr(3, 2, a.Off(0, 0).UpdateRows(3), tau, 8, w, 0)
	chkxer2("Zgeqr", err)

	//     Zgemqr
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "Zgemqr"
	// nb = 1
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Zgemqr('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Zgemqr(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgemqr(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgemqr(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, mn=0")
	err = golapack.Zgemqr(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, mn=0")
	err = golapack.Zgemqr(Right, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, mn): a.Rows=1, mn=2")
	err = golapack.Zgemqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Zgemqr(Right, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Zgemqr(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zgemqr(Left, NoTrans, 2, 1, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemqr", err)
	*errt = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=0, lw=0, lquery=false")
	err = golapack.Zgemqr(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(2), w, 0)
	chkxer2("Zgemqr", err)

	//     Zgelq
	*srnamt = "Zgelq"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgelq(-1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgelq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgelq(0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgelq", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgelq(2, 2, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Zgelq", err)
	*errt = fmt.Errorf("tsize < max(1, mb*m*nblcks+5) && (!lquery) && (!lminws): tsize=1, mb=1, m=2, nblcks=1, lquery=false, lminws=false")
	err = golapack.Zgelq(2, 3, a.Off(0, 0).UpdateRows(3), tau, 1, w, 1)
	chkxer2("Zgelq", err)
	*errt = fmt.Errorf("(lwork < max(1, m*mb)) && (!lquery) && (!lminws): lwork=0, m=2, mb=1, lquery=false, lminws=false")
	err = golapack.Zgelq(2, 3, a.Off(0, 0).UpdateRows(3), tau, 8, w, 0)
	chkxer2("Zgelq", err)

	//     Zgemlq
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "Zgemlq"
	// nb = 1
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Zgemlq('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Zgemlq(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgemlq(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgemlq(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, mn=0")
	err = golapack.Zgemlq(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, mn=0")
	err = golapack.Zgemlq(Right, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Zgemlq(Left, NoTrans, 2, 2, 2, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Zgemlq(Right, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Zgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zgemlq", err)
	*errt = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=0, lw=2, lquery=false")
	err = golapack.Zgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(2), w, 0)
	chkxer2("Zgemlq", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
