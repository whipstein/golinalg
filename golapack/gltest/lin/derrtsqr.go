package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrtsqr tests the error exits for the DOUBLE PRECISION routines
// that use the TSQR decomposition of a general matrix.
func derrtsqr(path string, _t *testing.T) {
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
	tau := vf(3)

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

	//     Error exits for TS factorization
	//
	//     Dgeqr
	*srnamt = "Dgeqr"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqr(-1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgeqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqr(0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgeqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqr(2, 1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgeqr", err)
	*errt = fmt.Errorf("tsize < max(1, nb*n*nblcks+5) && (!lquery) && (!lminws): tsize=1, n=2, nb=1, nblcks=1, lquery=false, lminws=false")
	err = golapack.Dgeqr(3, 2, a.Off(0, 0).UpdateRows(3), tau, 1, w, 1)
	chkxer2("Dgeqr", err)
	*errt = fmt.Errorf("(lwork < max(1, n*nb)) && (!lquery) && (!lminws): lwork=0, n=2, nb=1, lquery=false, lminws=false")
	err = golapack.Dgeqr(3, 2, a.Off(0, 0).UpdateRows(3), tau, 7, w, 0)
	chkxer2("Dgeqr", err)

	//     Dgemqr
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "Dgemqr"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dgemqr('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dgemqr(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgemqr(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgemqr(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, m=0, n=0")
	err = golapack.Dgemqr(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, m=0, n=0")
	err = golapack.Dgemqr(Right, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, mn): a.Rows=1, m=2, n=1")
	err = golapack.Dgemqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Dgemqr(Right, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Dgemqr(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dgemqr(Left, NoTrans, 2, 1, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=0, lw=0")
	err = golapack.Dgemqr(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(2), w, 0)
	chkxer2("Dgemqr", err)

	//     Dgelq
	*srnamt = "Dgelq"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgelq(-1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgelq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgelq(0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgelq", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgelq(2, 1, a.Off(0, 0).UpdateRows(1), tau, 1, w, 1)
	chkxer2("Dgelq", err)
	*errt = fmt.Errorf("tsize < max(1, mb*m*nblcks+5) && (!lquery) && (!lminws): tsize=1, m=2, n=3, mb=1, nb=3, lminws=false, lquery=false")
	err = golapack.Dgelq(2, 3, a.Off(0, 0).UpdateRows(3), tau, 1, w, 1)
	chkxer2("Dgelq", err)
	*errt = fmt.Errorf("(lwork < max(1, m*mb)) && (!lquery) && (!lminws): m=2, mb=1, lwork=0, lminws=false, lquery=false")
	err = golapack.Dgelq(2, 3, a.Off(0, 0).UpdateRows(3), tau, 7, w, 0)
	chkxer2("Dgelq", err)

	//     Dgemlq
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "Dgemlq"
	*errt = fmt.Errorf("!left && !right: side=Unrecognized: /")
	err = golapack.Dgemlq('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("!tran && !notran: trans=Unrecognized: /")
	err = golapack.Dgemlq(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgemlq(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgemlq(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, m=0, n=0")
	err = golapack.Dgemlq(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("k < 0 || k > mn: k=-1, m=0, n=0")
	err = golapack.Dgemlq(Right, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dgemlq(Left, NoTrans, 2, 3, 2, a.Off(0, 0).UpdateRows(1), tau, 1, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Dgemlq(Right, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("tsize < 5: tsize=0")
	err = golapack.Dgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, 0, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(1), tau, 6, c.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgemqr", err)
	*errt = fmt.Errorf("(lwork < max(1, lw)) && (!lquery): lwork=0, lw=2, lquery=false")
	err = golapack.Dgemlq(Left, NoTrans, 2, 2, 1, a.Off(0, 0).UpdateRows(2), tau, 6, c.Off(0, 0).UpdateRows(2), w, 0)
	chkxer2("Dgemqr", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		_t.Fail()
	}
}
