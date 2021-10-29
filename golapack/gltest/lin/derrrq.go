package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrrq tests the error exits for the DOUBLE PRECISION routines
// that use the RQ decomposition of a general matrix.
func derrrq(path string, t *testing.T) {
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

	//     Error exits for RQ factorization
	//     Dgerqf
	*srnamt = "Dgerqf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgerqf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgerqf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgerqf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgerqf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgerqf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 2)
	chkxer2("Dgerqf", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=1, m=2, lquery=false")
	err = golapack.Dgerqf(2, 1, a.Off(0, 0).UpdateRows(2), b, w, 1)
	chkxer2("Dgerqf", err)

	//     Dgerq2
	*srnamt = "Dgerq2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgerq2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgerq2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgerq2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgerq2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgerq2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgerq2", err)

	//     Dgerqs
	*srnamt = "Dgerqs"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = dgerqs(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=0, n=-1")
	err = dgerqs(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=2, n=1")
	err = dgerqs(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = dgerqs(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = dgerqs(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
	err = dgerqs(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = dgerqs(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgerqs", err)

	//     Dorgrq
	*srnamt = "Dorgrq"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorgrq(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("n < m: n=-1, m=0")
	err = golapack.Dorgrq(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("n < m: n=1, m=2")
	err = golapack.Dorgrq(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w, 2)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=-1, m=0")
	err = golapack.Dorgrq(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=2, m=1")
	err = golapack.Dorgrq(1, 2, 2, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorgrq(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Dorgrq", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=1, m=2, lquery=false")
	err = golapack.Dorgrq(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w, 1)
	chkxer2("Dorgrq", err)

	//     Dorgr2
	*srnamt = "Dorgr2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorgr2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgr2", err)
	*errt = fmt.Errorf("n < m: n=-1, m=0")
	err = golapack.Dorgr2(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgr2", err)
	*errt = fmt.Errorf("n < m: n=1, m=2")
	err = golapack.Dorgr2(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Dorgr2", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=-1, m=0")
	err = golapack.Dorgr2(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgr2", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=2, m=1")
	err = golapack.Dorgr2(1, 2, 2, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Dorgr2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorgr2(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgr2", err)

	//     Dormrq
	*srnamt = "Dormrq"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dormrq('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dormrq(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dormrq(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dormrq(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dormrq(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormrq(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormrq(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormrq(Left, NoTrans, 2, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormrq(Right, NoTrans, 1, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dormrq(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormrq(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormrq", err)
	*errt = fmt.Errorf("lwork < nw && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormrq(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormrq", err)

	//     Dormr2
	*srnamt = "Dormr2"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dormr2('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dormr2(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dormr2(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dormr2(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dormr2(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormr2(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormr2(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormr2(Left, NoTrans, 2, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormr2(Right, NoTrans, 1, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dormr2(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dormr2", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
