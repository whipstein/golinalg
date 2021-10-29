package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrql tests the error exits for the DOUBLE PRECISION routines
// that use the QL decomposition of a general matrix.
func derrql(path string, t *testing.T) {
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

	//     Error exits for QL factorization
	//
	//     Dgeqlf
	*srnamt = "Dgeqlf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeqlf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqlf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeqlf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqlf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeqlf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqlf", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Dgeqlf(1, 2, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgeqlf", err)

	//     Dgeql2
	*srnamt = "Dgeql2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgeql2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeql2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgeql2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeql2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgeql2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgeql2", err)

	//     Dgeqls
	*srnamt = "Dgeqls"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = dgeqls(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=0, n=-1")
	err = dgeqls(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("n < 0 || n > m: m=1, n=2")
	err = dgeqls(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = dgeqls(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = dgeqls(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("b.Rows < max(1, m): b.Rows=1, m=2")
	err = dgeqls(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = dgeqls(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgeqls", err)

	//     Dorgql
	*srnamt = "Dorgql"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorgql(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Dorgql(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Dorgql(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Dorgql(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Dorgql(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorgql(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgql", err)
	*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
	err = golapack.Dorgql(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w, 1)
	chkxer2("Dorgql", err)

	//     Dorg2l
	*srnamt = "Dorg2l"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorg2l(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2l", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=-1, m=0")
	err = golapack.Dorg2l(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2l", err)
	*errt = fmt.Errorf("n < 0 || n > m: n=2, m=1")
	err = golapack.Dorg2l(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2l", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=-1, n=0")
	err = golapack.Dorg2l(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2l", err)
	*errt = fmt.Errorf("k < 0 || k > n: k=2, n=1")
	err = golapack.Dorg2l(2, 1, 2, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Dorg2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorg2l(2, 1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorg2l", err)

	//     Dormql
	*srnamt = "Dormql"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dormql('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dormql(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dormql(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dormql(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dormql(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormql(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormql(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dormql(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dormql(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dormql(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormql(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormql", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormql(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormql", err)

	//     Dorm2l
	*srnamt = "Dorm2l"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dorm2l('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dorm2l(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorm2l(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dorm2l(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dorm2l(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorm2l(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorm2l(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dorm2l(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("a.Rows < max(1, nq): a.Rows=1, nq=2")
	err = golapack.Dorm2l(Right, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dorm2l(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorm2l", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
