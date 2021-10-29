package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrge tests the error exits for the COMPLEX*16 routines
// for general matrices.
func zerrge(path string, t *testing.T) {
	var anrm float64
	var i, j, nmax int
	var err error

	nmax = 4
	b := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	ip := make([]int, 4)
	a := cmf(4, 4, opts)
	af := cmf(4, 4, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
			af.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
		}
		b.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
		ip[j-1] = j
	}
	*ok = true

	//     Test error exits of the routines that use the LU decomposition
	//     of a general matrix.
	if c2 == "ge" {
		//        Zgetrf
		*srnamt = "Zgetrf"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgetrf(-1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgetrf(0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgetrf(2, 1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetrf", err)

		//        Zgetf2
		*srnamt = "Zgetf2"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgetf2(-1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgetf2(0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgetf2(2, 1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgetf2", err)

		//        Zgetri
		*srnamt = "Zgetri"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgetri(-1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zgetri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zgetri(2, a.Off(0, 0).UpdateRows(1), &ip, w, 2)
		chkxer2("Zgetri", err)
		*errt = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=1, n=2, lquery=false")
		_, err = golapack.Zgetri(2, a.Off(0, 0).UpdateRows(2), &ip, w, 1)
		chkxer2("Zgetri", err)

		//        Zgetrs
		*srnamt = "Zgetrs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgetrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgetrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgetrs(NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgetrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgetrs(NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgetrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zgetrs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("Zgetrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zgetrs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("Zgetrs", err)

		//        Zgerfs
		*srnamt = "Zgerfs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgerfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgerfs(NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgerfs(NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Zgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zgerfs(NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgerfs", err)

		//        Zgecon
		*srnamt = "Zgecon"
		// *errt = fmt.Errorf("anorm < zero: anorm=%v", anorm)
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Zgecon('/', 0, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zgecon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgecon('1', -1, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zgecon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zgecon('1', 2, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zgecon", err)

		//        Zgeequ
		*srnamt = "Zgeequ"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, _, _, err = golapack.Zgeequ(-1, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgeequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Zgeequ(0, -1, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgeequ", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, _, _, err = golapack.Zgeequ(2, 2, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgeequ", err)

		//     Test error exits of the routines that use the LU decomposition
		//     of a general band matrix.
	} else if c2 == "gb" {
		//        Zgbtrf
		*srnamt = "Zgbtrf"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgbtrf(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgbtrf(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtrf", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Zgbtrf(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtrf", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Zgbtrf(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtrf", err)
		*errt = fmt.Errorf("ab.Rows < kl+kv+1: ab.Rows=3, kl=1, kv=2")
		_, err = golapack.Zgbtrf(2, 2, 1, 1, a.Off(0, 0).UpdateRows(3), &ip)
		chkxer2("Zgbtrf", err)

		//        Zgbtf2
		*srnamt = "Zgbtf2"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgbtf2(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgbtf2(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtf2", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Zgbtf2(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtf2", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Zgbtf2(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zgbtf2", err)
		*errt = fmt.Errorf("ab.Rows < kl+kv+1: ab.Rows=3, kl=1, kv=2")
		_, err = golapack.Zgbtf2(2, 2, 1, 1, a.Off(0, 0).UpdateRows(3), &ip)
		chkxer2("Zgbtf2", err)

		//        Zgbtrs
		*srnamt = "Zgbtrs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgbtrs('/', 0, 0, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgbtrs(NoTrans, -1, 0, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		err = golapack.Zgbtrs(NoTrans, 1, -1, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		err = golapack.Zgbtrs(NoTrans, 1, 0, -1, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgbtrs(NoTrans, 1, 0, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("ab.Rows < (2*kl + ku + 1): ab.Rows=3, kl=1, ku=1")
		err = golapack.Zgbtrs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(3), &ip, b.CMatrix(2, opts))
		chkxer2("Zgbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zgbtrs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbtrs", err)

		//        Zgbrfs
		*srnamt = "Zgbrfs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgbrfs('/', 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgbrfs(NoTrans, -1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		err = golapack.Zgbrfs(NoTrans, 1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		err = golapack.Zgbrfs(NoTrans, 1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgbrfs(NoTrans, 1, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		err = golapack.Zgbrfs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(4), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("afb.Rows < 2*kl+ku+1: afb.Rows=3, kl=1, ku=1")
		err = golapack.Zgbrfs(NoTrans, 2, 1, 1, 1, a.Off(0, 0).UpdateRows(3), af.Off(0, 0).UpdateRows(3), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zgbrfs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zgbrfs(NoTrans, 2, 0, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zgbrfs", err)

		//        Zgbcon
		*srnamt = "Zgbcon"
		// *errt = fmt.Errorf("anorm < zero: anorm=%v", anorm)
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Zgbcon('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, r)
		chkxer2("Zgbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgbcon('1', -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, r)
		chkxer2("Zgbcon", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Zgbcon('1', 1, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, r)
		chkxer2("Zgbcon", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Zgbcon('1', 1, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, r)
		chkxer2("Zgbcon", err)
		*errt = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Zgbcon('1', 2, 1, 1, a.Off(0, 0).UpdateRows(3), &ip, anrm, w, r)
		chkxer2("Zgbcon", err)

		//        Zgbequ
		*srnamt = "Zgbequ"
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, _, _, err = golapack.Zgbequ(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgbequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Zgbequ(0, -1, 0, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgbequ", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, _, _, _, err = golapack.Zgbequ(1, 1, -1, 0, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgbequ", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, _, _, _, err = golapack.Zgbequ(1, 1, 0, -1, a.Off(0, 0).UpdateRows(1), r1, r2)
		chkxer2("Zgbequ", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		_, _, _, _, err = golapack.Zgbequ(2, 2, 1, 1, a.Off(0, 0).UpdateRows(2), r1, r2)
		chkxer2("Zgbequ", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
