package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrqr tests the error exits for the COMPLEX*16 routines
// that use the QR decomposition of a general matrix.
func zerrqr(path string, t *testing.T) {
	var i, j, nmax int
	var err error

	b := cvf(2)
	w := cvf(2)
	x := cvf(2)
	a := cmf(2, 2, opts)
	af := cmf(2, 2, opts)

	nmax = 2
	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
			af.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
		}
		b.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QR factorization
	//
	//     Zgeqrf
	*srnamt = "Zgeqrf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqrf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqrf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqrf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrf", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Zgeqrf(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrf", err)

	//     Zgeqrfp
	*srnamt = "Zgeqrfp"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqrfp(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrfp", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqrfp(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrfp", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqrfp(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrfp", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Zgeqrfp(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqrfp", err)

	//     Zgeqr2
	*srnamt = "Zgeqr2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqr2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqr2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqr2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2", err)

	//     Zgeqr2p
	*srnamt = "Zgeqr2p"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqr2p(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2p", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqr2p(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2p", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqr2p(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeqr2p", err)

	//     zgeqrs
	*srnamt = "zgeqrs"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = zgeqrs(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = zgeqrs(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = zgeqrs(1, 2, 0, a.Off(0, 0).UpdateRows(2), x, b.CMatrix(2, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = zgeqrs(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = zgeqrs(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(2, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = zgeqrs(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = zgeqrs(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgeqrs", err)

	//     Zungqr
	*srnamt = "Zungqr"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zungqr(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=0, n=-1")
	err = golapack.Zungqr(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=1, n=2")
	err = golapack.Zungqr(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 2), 2)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("k < 0 || k > n: n=0, k=-1")
	err = golapack.Zungqr(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("k < 0 || k > n: n=1, k=2")
	err = golapack.Zungqr(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zungqr(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 2), 2)
	chkxer2("Zungqr", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Zungqr(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w.Off(0, 1), 1)
	chkxer2("Zungqr", err)

	//     Zung2r
	*srnamt = "Zung2r"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zung2r(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2r", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Zung2r(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2r", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Zung2r(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2r", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Zung2r(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2r", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Zung2r(2, 1, 2, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Zung2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zung2r(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2r", err)

	//     Zunmqr
	*srnamt = "Zunmqr"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunmqr('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunmqr(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunmqr(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunmqr(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunmqr(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmqr(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmqr(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunmqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunmqr(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunmqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmqr(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmqr(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
	chkxer2("Zunmqr", err)

	//     Zunm2r
	*srnamt = "Zunm2r"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunm2r('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunm2r(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunm2r(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunm2r(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunm2r(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunm2r(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunm2r(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunm2r(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunm2r(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunm2r(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2r", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
