package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrlq tests the error exits for the COMPLEX*16 routines
// that use the LQ decomposition of a general matrix.
func zerrlq(path string, t *testing.T) {
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

	//     Error exits for LQ factorization
	//
	//     Zgelqf
	*srnamt = "Zgelqf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgelqf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgelqf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgelqf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgelqf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgelqf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 2)
	chkxer2("Zgelqf", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=1, m=2, lquery=false")
	err = golapack.Zgelqf(2, 1, a.Off(0, 0).UpdateRows(2), b, w, 1)
	chkxer2("Zgelqf", err)

	//     Zgelq2
	*srnamt = "Zgelq2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgelq2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgelq2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgelq2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgelq2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgelq2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgelq2", err)

	//     zgelqs
	*srnamt = "zgelqs"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = zgelqs(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=0, n=-1")
	err = zgelqs(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=2, n=1")
	err = zgelqs(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = zgelqs(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = zgelqs(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(2, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
	err = zgelqs(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = zgelqs(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w.Off(0, 1), 1)
	chkxer2("zgelqs", err)

	//     Zunglq
	*srnamt = "Zunglq"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunglq(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("n < m: m=0, n=-1")
	err = golapack.Zunglq(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("n < m: m=2, n=1")
	err = golapack.Zunglq(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w.Off(0, 2), 2)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("k < 0 || k > m: m=0, k=-1")
	err = golapack.Zunglq(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("k < 0 || k > m: m=1, k=2")
	err = golapack.Zunglq(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 1), 1)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zunglq(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w.Off(0, 2), 2)
	chkxer2("Zunglq", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=1, m=2, lquery=false")
	err = golapack.Zunglq(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w.Off(0, 1), 1)
	chkxer2("Zunglq", err)

	//     Zungl2
	*srnamt = "Zungl2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zungl2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zungl2", err)
	*errt = fmt.Errorf("n < m: m=0, n=-1")
	err = golapack.Zungl2(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zungl2", err)
	*errt = fmt.Errorf("n < m: m=2, n=1")
	err = golapack.Zungl2(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Zungl2", err)
	*errt = fmt.Errorf("k < 0 || k > m: m=0, k=-1")
	err = golapack.Zungl2(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zungl2", err)
	*errt = fmt.Errorf("k < 0 || k > m: m=1, k=2")
	err = golapack.Zungl2(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zungl2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zungl2(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zungl2", err)

	//     Zunmlq
	*srnamt = "Zunmlq"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunmlq('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunmlq(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunmlq(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunmlq(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunmlq(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmlq(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmlq(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Zunmlq(Left, NoTrans, 2, 0, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Zunmlq(Right, NoTrans, 0, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunmlq(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmlq(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmlq(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w.Off(0, 1), 1)
	chkxer2("Zunmlq", err)

	//     Zunml2
	*srnamt = "Zunml2"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunml2('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunml2(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunml2(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunml2(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunml2(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunml2(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunml2(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Zunml2(Left, NoTrans, 2, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Zunml2(Right, NoTrans, 1, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunml2(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunml2", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
