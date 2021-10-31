package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrgt tests the error exits for the DOUBLE PRECISION tridiagonal
// routines.
func derrgt(path string, t *testing.T) {
	var anorm float64
	var err error

	b := mf(2, 1, opts)
	c := vf(2)
	cf := vf(2)
	d := vf(2)
	df := vf(2)
	e := vf(2)
	ef := vf(2)
	f := vf(2)
	r1 := vf(2)
	r2 := vf(2)
	w := vf(2)
	x := mf(2, 1, opts)
	ip := make([]int, 2)
	iw := make([]int, 2)
	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	d.Set(0, 1.)
	d.Set(1, 2.)
	df.Set(0, 1.)
	df.Set(1, 2.)
	e.Set(0, 3.)
	e.Set(1, 4.)
	ef.Set(0, 3.)
	ef.Set(1, 4.)
	anorm = 1.0
	(*ok) = true

	if c2 == "gt" {
		//        Test error exits for the general tridiagonal routines.
		//
		//        Dgttrf
		*srnamt = "Dgttrf"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgttrf(-1, c, d, e, f, &ip)
		chkxer2("Dgttrf", err)

		//        Dgttrs
		*srnamt = "Dgttrs"
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgttrs('/', 0, 0, c, d, e, f, ip, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dgttrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgttrs(NoTrans, -1, 0, c, d, e, f, ip, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dgttrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgttrs(NoTrans, 0, -1, c, d, e, f, ip, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dgttrs", err)
		*errt = fmt.Errorf("b.Rows < max(n, 1): b.Rows=1, n=2")
		err = golapack.Dgttrs(NoTrans, 2, 1, c, d, e, f, ip, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dgttrs", err)

		//        Dgtrfs
		*srnamt = "Dgtrfs"
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		err = golapack.Dgtrfs('/', 0, 0, c, d, e, cf, df, ef, f, ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dgtrfs(NoTrans, -1, 0, c, d, e, cf, df, ef, f, ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dgtrfs(NoTrans, 0, -1, c, d, e, cf, df, ef, f, ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dgtrfs(NoTrans, 2, 1, c, d, e, cf, df, ef, f, ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgtrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dgtrfs(NoTrans, 2, 1, c, d, e, cf, df, ef, f, ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtrfs", err)

		//        Dgtcon
		*srnamt = "Dgtcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm=/")
		_, err = golapack.Dgtcon('/', 0, c, d, e, f, ip, anorm, w, &iw)
		chkxer2("Dgtcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgtcon('I', -1, c, d, e, f, ip, anorm, w, &iw)
		chkxer2("Dgtcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Dgtcon('I', 0, c, d, e, f, ip, -anorm, w, &iw)
		chkxer2("Dgtcon", err)

	} else if c2 == "pt" {
		//        Test error exits for the positive definite tridiagonal
		//        routines.
		//
		//        Dpttrf
		*srnamt = "Dpttrf"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpttrf(-1, d, e)
		chkxer2("Dpttrf", err)

		//        Dpttrs
		*srnamt = "Dpttrs"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpttrs(-1, 0, d, e, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dpttrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpttrs(0, -1, d, e, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dpttrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpttrs(2, 1, d, e, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dpttrs", err)

		//        Dptrfs
		*srnamt = "Dptrfs"
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dptrfs(-1, 0, d, e, df, ef, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dptrfs(0, -1, d, e, df, ef, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dptrfs(2, 1, d, e, df, ef, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w)
		chkxer2("Dptrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dptrfs(2, 1, d, e, df, ef, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptrfs", err)

		//        Dptcon
		*srnamt = "Dptcon"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dptcon(-1, d, e, anorm, w)
		chkxer2("Dptcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Dptcon(0, d, e, -anorm, w)
		chkxer2("Dptcon", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
