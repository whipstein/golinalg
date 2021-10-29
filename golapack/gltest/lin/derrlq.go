package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrlq tests the error exits for the DOUBLE PRECISION routines
// that use the LQ decomposition of a general matrix.
func derrlq(path string, t *testing.T) {
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

	//     Error exits for LQ factorization
	//
	//     Dgelqf
	*srnamt = "Dgelqf"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgelqf(-1, 0, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgelqf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgelqf(0, -1, a.Off(0, 0).UpdateRows(1), b, w, 1)
	chkxer2("Dgelqf", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgelqf(2, 1, a.Off(0, 0).UpdateRows(1), b, w, 2)
	chkxer2("Dgelqf", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: m=2, lwork=1, lquery=false")
	err = golapack.Dgelqf(2, 1, a.Off(0, 0).UpdateRows(2), b, w, 1)
	chkxer2("Dgelqf", err)

	//     Dgelq2
	*srnamt = "Dgelq2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dgelq2(-1, 0, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgelq2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dgelq2(0, -1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgelq2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dgelq2(2, 1, a.Off(0, 0).UpdateRows(1), b, w)
	chkxer2("Dgelq2", err)

	//     Dgelqs
	*srnamt = "Dgelqs"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = dgelqs(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=0, n=-1")
	err = dgelqs(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("n < 0 || m > n: m=2, n=1")
	err = dgelqs(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
	err = dgelqs(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = dgelqs(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
	err = dgelqs(1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)
	*errt = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=1, nrhs=2, m=1, n=1")
	err = dgelqs(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dgelqs", err)

	//     Dorglq
	*srnamt = "Dorglq"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorglq(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("n < m: n=-1, m=0")
	err = golapack.Dorglq(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("n < m: n=1, m=2")
	err = golapack.Dorglq(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w, 2)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=-1, m=0")
	err = golapack.Dorglq(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=2, m=1")
	err = golapack.Dorglq(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w, 1)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorglq(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w, 2)
	chkxer2("Dorglq", err)
	*errt = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=1, m=2, lquery=false")
	err = golapack.Dorglq(2, 2, 0, a.Off(0, 0).UpdateRows(2), x, w, 1)
	chkxer2("Dorglq", err)

	//     Dorgl2
	*srnamt = "Dorgl2"
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorgl2(-1, 0, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgl2", err)
	*errt = fmt.Errorf("n < m: n=-1, m=0")
	err = golapack.Dorgl2(0, -1, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgl2", err)
	*errt = fmt.Errorf("n < m: n=1, m=2")
	err = golapack.Dorgl2(2, 1, 0, a.Off(0, 0).UpdateRows(2), x, w)
	chkxer2("Dorgl2", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=-1, m=0")
	err = golapack.Dorgl2(0, 0, -1, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgl2", err)
	*errt = fmt.Errorf("k < 0 || k > m: k=2, m=1")
	err = golapack.Dorgl2(1, 1, 2, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgl2", err)
	*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
	err = golapack.Dorgl2(2, 2, 0, a.Off(0, 0).UpdateRows(1), x, w)
	chkxer2("Dorgl2", err)

	//     Dormlq
	*srnamt = "Dormlq"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dormlq('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dormlq(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dormlq(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dormlq(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dormlq(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormlq(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dormlq(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormlq(Left, NoTrans, 2, 0, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dormlq(Right, NoTrans, 0, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dormlq(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormlq(Left, NoTrans, 1, 2, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w, 1)
	chkxer2("Dormlq", err)
	*errt = fmt.Errorf("lwork < max(1, nw) && !lquery: lwork=1, nw=2, lquery=false")
	err = golapack.Dormlq(Right, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w, 1)
	chkxer2("Dormlq", err)

	//     Dorml2
	*srnamt = "Dorml2"
	*errt = fmt.Errorf("!left && side != Right: side=Unrecognized: /")
	err = golapack.Dorml2('/', NoTrans, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("!notran && trans != Trans: trans=Unrecognized: /")
	err = golapack.Dorml2(Left, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("m < 0: m=-1")
	err = golapack.Dorml2(Left, NoTrans, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	err = golapack.Dorml2(Left, NoTrans, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=-1, nq=0")
	err = golapack.Dorml2(Left, NoTrans, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorml2(Left, NoTrans, 0, 1, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("k < 0 || k > nq: k=1, nq=0")
	err = golapack.Dorml2(Right, NoTrans, 1, 0, 1, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dorml2(Left, NoTrans, 2, 1, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(2), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("a.Rows < max(1, k): a.Rows=1, k=2")
	err = golapack.Dorml2(Right, NoTrans, 1, 2, 2, a.Off(0, 0).UpdateRows(1), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)
	*errt = fmt.Errorf("c.Rows < max(1, m): c.Rows=1, m=2")
	err = golapack.Dorml2(Left, NoTrans, 2, 1, 0, a.Off(0, 0).UpdateRows(2), x, af.Off(0, 0).UpdateRows(1), w)
	chkxer2("Dorml2", err)

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
