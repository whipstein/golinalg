package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrgt tests the error exits for the COMPLEX*16 tridiagonal
// routines.
func zerrgt(path string, t *testing.T) {
	var anorm float64
	var i, nmax int
	var err error

	b := cvf(2)
	dl := cvf(2)
	dlf := cvf(2)
	du := cvf(2)
	du2 := cvf(2)
	duf := cvf(2)
	e := cvf(2)
	ef := cvf(2)
	w := cvf(2)
	x := cvf(2)
	d := vf(2)
	df := vf(2)
	r1 := vf(2)
	r2 := vf(2)
	rw := vf(2)
	ip := make([]int, 2)

	nmax = 2

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]
	for i = 1; i <= nmax; i++ {
		d.Set(i-1, 1.)
		e.Set(i-1, 2.)
		dl.Set(i-1, 3.)
		du.Set(i-1, 4.)
	}
	anorm = 1.0
	*ok = true

	if c2 == "gt" {
		//        Test error exits for the general tridiagonal routines.
		//
		//        Zgttrf
		*srnamt = "Zgttrf"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgttrf(-1, dl, e, du, du2, &ip)
		chkxer2("Zgttrf", err)

		//        Zgttrs
		*srnamt = "Zgttrs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgttrs('/', 0, 0, dl, e, du, du2, &ip, x.CMatrix(1, opts))
		chkxer2("Zgttrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgttrs(NoTrans, -1, 0, dl, e, du, du2, &ip, x.CMatrix(1, opts))
		chkxer2("Zgttrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgttrs(NoTrans, 0, -1, dl, e, du, du2, &ip, x.CMatrix(1, opts))
		chkxer2("Zgttrs", err)
		*errt = fmt.Errorf("b.Rows < max(n, 1): b.Rows=1, n=2")
		err = golapack.Zgttrs(NoTrans, 2, 1, dl, e, du, du2, &ip, x.CMatrix(1, opts))
		chkxer2("Zgttrs", err)

		//        Zgtrfs
		*srnamt = "Zgtrfs"
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Zgtrfs('/', 0, 0, dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zgtrfs(NoTrans, -1, 0, dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zgtrfs(NoTrans, 0, -1, dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zgtrfs(NoTrans, 2, 1, dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgtrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zgtrfs(NoTrans, 2, 1, dl, e, du, dlf, ef, duf, du2, &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtrfs", err)

		//        Zgtcon
		*srnamt = "Zgtcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Zgtcon('/', 0, dl, e, du, du2, &ip, anorm, w)
		chkxer2("Zgtcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgtcon('I', -1, dl, e, du, du2, &ip, anorm, w)
		chkxer2("Zgtcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zgtcon('I', 0, dl, e, du, du2, &ip, -anorm, w)
		chkxer2("Zgtcon", err)

	} else if c2 == "pt" {
		//        Test error exits for the positive definite tridiagonal
		//        routines.
		//
		//        Zpttrf
		*srnamt = "Zpttrf"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpttrf(-1, d, e)
		chkxer2("Zpttrf", err)

		//        Zpttrs
		*srnamt = "Zpttrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpttrs('/', 1, 0, d, e, x.CMatrix(1, opts))
		chkxer2("Zpttrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpttrs(Upper, -1, 0, d, e, x.CMatrix(1, opts))
		chkxer2("Zpttrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpttrs(Upper, 0, -1, d, e, x.CMatrix(1, opts))
		chkxer2("Zpttrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpttrs(Upper, 2, 1, d, e, x.CMatrix(1, opts))
		chkxer2("Zpttrs", err)

		//        Zptrfs
		*srnamt = "Zptrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zptrfs('/', 1, 0, d, e, df, ef, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zptrfs(Upper, -1, 0, d, e, df, ef, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zptrfs(Upper, 0, -1, d, e, df, ef, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zptrfs(Upper, 2, 1, d, e, df, ef, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zptrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zptrfs(Upper, 2, 1, d, e, df, ef, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptrfs", err)

		//        Zptcon
		*srnamt = "Zptcon"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zptcon(-1, d, e, anorm, rw)
		chkxer2("Zptcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zptcon(0, d, e, -anorm, rw)
		chkxer2("Zptcon", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
