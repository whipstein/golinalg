package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrql tests the error exits for the COMPLEX*16 routines
// that use the QL decomposition of a general matrix.
func zerrql(path string, t *testing.T) {
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

	//     Error exits for QL factorization
	//
	//     Zgeqlf
	*srnamt = "Zgeqlf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeqlf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqlf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeqlf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqlf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeqlf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqlf", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Zgeqlf(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Zgeqlf", err)

	//     Zgeql2
	*srnamt = "Zgeql2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zgeql2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeql2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zgeql2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeql2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zgeql2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Zgeql2", err)

	//     zgeqls
	*srnamt = "zgeqls"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = zgeqls(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = zgeqls(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = zgeqls(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = zgeqls(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = zgeqls(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(2, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = zgeqls(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = zgeqls(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, b.CMatrix(1, opts), w, 1)
	chkxer2("zgeqls", err)

	//     Zungql
	*srnamt = "Zungql"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zungql(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=0, n=-1")
	err = golapack.Zungql(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=1, n=2")
	err = golapack.Zungql(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("k < 0 || k > n: n=0, k=-1")
	err = golapack.Zungql(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("k < 0 || k > n: n=1, k=2")
	err = golapack.Zungql(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zungql(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Zungql", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Zungql(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w, 1)
	chkxer2("Zungql", err)

	//     Zung2l
	*srnamt = "Zung2l"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zung2l(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2l", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Zung2l(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2l", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Zung2l(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2l", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Zung2l(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2l", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Zung2l(2, 1, 2, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Zung2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Zung2l(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Zung2l", err)

	//     Zunmql
	*srnamt = "Zunmql"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunmql('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunmql(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunmql(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunmql(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunmql(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmql(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunmql(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunmql(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunmql(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunmql(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmql(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Zunmql", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Zunmql(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Zunmql", err)

	//     Zunm2l
	*srnamt = "Zunm2l"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Zunm2l('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("!notran && trans != ConjTrans: trans=Unrecognized: /")
	err = golapack.Zunm2l(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Zunm2l(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Zunm2l(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Zunm2l(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunm2l(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Zunm2l(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunm2l(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Zunm2l(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Zunm2l(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Zunm2l", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
