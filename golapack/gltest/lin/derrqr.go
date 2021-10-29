package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrqr tests the error exits for the DOUBLE PRECISION routines
// that use the QR decomposition of a general matrix.
func derrqr(path string, t *testing.T) {
	var i, j, nmax int
	var err error
	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	af := mf(2, 2, opts)
	b := vf(2)
	w := vf(2)
	x := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QR factorization
	//     Dgeqrf
	*srnamt = "Dgeqrf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqrf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqrf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqrf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrf", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Dgeqrf(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrf", err)

	//     Dgeqrfp
	*srnamt = "Dgeqrfp"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqrfp(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrfp", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqrfp(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrfp", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqrfp(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrfp", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Dgeqrfp(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqrfp", err)

	//     Dgeqr2
	*srnamt = "Dgeqr2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqr2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqr2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqr2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2", err)

	//     Dgeqr2p
	*srnamt = "Dgeqr2p"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqr2p(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2p", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqr2p(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2p", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqr2p(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeqr2p", err)

	//     Dgeqrs
	*srnamt = "Dgeqrs"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = dgeqrs(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=0, n=-1")
	err = dgeqrs(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=1, n=2")
	err = dgeqrs(1, 2, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("nrhs < 0: nrh=-1")
	err = dgeqrs(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = dgeqrs(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = dgeqrs(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqrs", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = dgeqrs(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqrs", err)

	//     Dorgqr
	*srnamt = "Dorgqr"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorgqr(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Dorgqr(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Dorgqr(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Dorgqr(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Dorgqr(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorgqr(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Dorgqr", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Dorgqr(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w, 1)
	chkxer2("Dorgqr", err)

	//     Dorg2r
	*srnamt = "Dorg2r"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorg2r(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2r", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Dorg2r(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2r", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Dorg2r(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2r", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Dorg2r(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2r", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Dorg2r(2, 1, 2, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Dorg2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorg2r(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2r", err)

	//     Dormqr
	*srnamt = "Dormqr"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dormqr('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dormqr(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dormqr(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dormqr(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dormqr(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormqr(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormqr(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dormqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dormqr(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dormqr(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormqr(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormqr", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormqr(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormqr", err)

	//     Dorm2r
	*srnamt = "Dorm2r"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dorm2r('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dorm2r(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorm2r(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dorm2r(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dorm2r(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorm2r(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorm2r(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dorm2r(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dorm2r(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dorm2r(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2r", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
