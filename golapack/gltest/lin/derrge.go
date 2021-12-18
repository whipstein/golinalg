package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrge tests the error exits for the DOUBLE PRECISION routines
// for general matrices.
func derrge(path string, t *testing.T) {
	var anrm float64
	var i, j, lw, nmax int
	var err error
	ip := make([]int, 4)
	iw := make([]int, 4)
	errt := &gltest.Common.Infoc.Errt
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4
	lw = 3 * nmax

	a := mf(4, 4, opts)
	af := mf(4, 4, opts)
	b := mf(4, 1, opts)
	w := mf(lw, 1, opts)
	x := mf(4, 1, opts)
	r1 := vf(4)
	r2 := vf(4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.SetIdx(j-1, 0.)
		x.SetIdx(j-1, 0.)
		ip[j-1] = j
		iw[j-1] = j
	}
	*lerr = false
	*ok = true

	if c2 == "ge" {
		//        Test error exits of the routines that use the LU decomposition
		//        of a general matrix.
		//
		//        Dgetrf
		*srnamt = "Dgetrf"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgetrf(-1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgetrf(0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgetrf(2, 1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetrf", err)

		//        Dgetf2
		*srnamt = "Dgetf2"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgetf2(-1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgetf2(0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgetf2(2, 1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgetf2", err)

		//        Dgetri
		*srnamt = "Dgetri"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgetri(-1, a.Off(0, 0).UpdateRows(1), ip, w)
		chkxer2("Dgetri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dgetri(2, a.Off(0, 0).UpdateRows(1), ip, w)
		chkxer2("Dgetri", err)

		//        Dgetrs
		*srnamt = "Dgetrs"
		*infot = 1
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgetrs('/', 0, 0, a, ip, b)
		chkxer2("Dgetrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgetrs(NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgetrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgetrs(NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgetrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dgetrs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("Dgetrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dgetrs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgetrs", err)

		//        Dgerfs
		*srnamt = "Dgerfs"
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgerfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgerfs(NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgerfs(NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Dgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgerfs", err)

		//        Dgecon
		*srnamt = "Dgecon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Dgecon('/', 0, a.Off(0, 0).UpdateRows(1), anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgecon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgecon('1', -1, a.Off(0, 0).UpdateRows(1), anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgecon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dgecon('1', 2, a.Off(0, 0).UpdateRows(1), anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgecon", err)

		//        Dgeequ
		*srnamt = "Dgeequ"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, _, _, err = golapack.Dgeequ(-1, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgeequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Dgeequ(0, -1, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgeequ", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, _, _, err = golapack.Dgeequ(2, 2, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgeequ", err)

	} else if c2 == "gb" {
		//        Test error exits of the routines that use the LU decomposition
		//        of a general band matrix.
		//
		//        Dgbtrf
		*srnamt = "Dgbtrf"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgbtrf(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgbtrf(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtrf", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Dgbtrf(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtrf", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Dgbtrf(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtrf", err)
		*errt = fmt.Errorf("ab.Rows < kl+kv+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Dgbtrf(2, 2, 1, 1, a.Off(0, 0).UpdateRows(3), &ip)
		chkxer2("Dgbtrf", err)

		//        Dgbtf2
		*srnamt = "Dgbtf2"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgbtf2(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgbtf2(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtf2", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Dgbtf2(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtf2", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Dgbtf2(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dgbtf2", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+kl+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Dgbtf2(2, 2, 1, 1, a.Off(0, 0).UpdateRows(3), &ip)
		chkxer2("Dgbtf2", err)

		//        Dgbtrs
		*srnamt = "Dgbtrs"
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgbtrs('/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgbtrs(NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		err = golapack.Dgbtrs(NoTrans, 1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		err = golapack.Dgbtrs(NoTrans, 1, 0, -1, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgbtrs(NoTrans, 1, 0, 0, -1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("ab.Rows < (2*kl + ku + 1): ab.Rows=3, kl=1, ku=1")
		err = golapack.Dgbtrs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(3), ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("Dgbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dgbtrs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbtrs", err)

		//        Dgbrfs
		*srnamt = "Dgbrfs"
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgbrfs('/', 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgbrfs(NoTrans, -1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		err = golapack.Dgbrfs(NoTrans, 1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		err = golapack.Dgbrfs(NoTrans, 1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgbrfs(NoTrans, 1, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		err = golapack.Dgbrfs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(4), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("afb.Rows < 2*kl+ku+1: afb.Rows=3, kl=1, ku=1")
		err = golapack.Dgbrfs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(3), af.Off(0, 0).UpdateRows(3), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dgbrfs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dgbrfs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbrfs", err)

		//        Dgbcon
		*srnamt = "Dgbcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Dgbcon('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), ip, anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgbcon('1', -1, 0, 0, a.Off(0, 0).UpdateRows(1), ip, anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbcon", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Dgbcon('1', 1, -1, 0, a.Off(0, 0).UpdateRows(1), ip, anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbcon", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Dgbcon('1', 1, 0, -1, a.Off(0, 0).UpdateRows(1), ip, anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbcon", err)
		*errt = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Dgbcon('1', 2, 1, 1, a.Off(0, 0).UpdateRows(3), ip, anrm, w.OffIdx(0).Vector(), &iw)
		chkxer2("Dgbcon", err)

		//        Dgbequ
		*srnamt = "Dgbequ"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, _, _, err = golapack.Dgbequ(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgbequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Dgbequ(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgbequ", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, _, _, _, err = golapack.Dgbequ(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgbequ", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, _, _, _, err = golapack.Dgbequ(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Dgbequ", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		_, _, _, _, err = golapack.Dgbequ(2, 2, 1, 1, a.Off(0, 0).UpdateRows(2), r1, r2)
		chkxer2("Dgbequ", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
