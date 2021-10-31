package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrtr tests the error exits for the DOUBLE PRECISION triangular
// routines.
func derrtr(path string, t *testing.T) {
	var scale float64
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	a := mf(2, 2, opts)
	ap := vf(2 * 2)
	b := mf(2, 1, opts)
	r1 := vf(2)
	r2 := vf(2)
	w := vf(2)
	x := mf(2, 1, opts)
	iw := make([]int, 2)
	a.Set(0, 0, 1.)
	a.Set(0, 1, 2.)
	a.Set(1, 1, 3.)
	a.Set(1, 0, 4.)
	(*ok) = true

	if c2 == "tr" {
		//        Test error exits for the general triangular routines.
		//
		//        Dtrtri
		*srnamt = "Dtrtri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtrtri('/', NonUnit, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtri", err)
		*errt = fmt.Errorf("!diag.IsValid(): diag=Unrecognized: /")
		_, err = golapack.Dtrtri(Upper, '/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtrtri(Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dtrtri(Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtri", err)

		//        Dtrti2
		*srnamt = "Dtrti2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dtrti2('/', NonUnit, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrti2", err)
		*errt = fmt.Errorf("!diag.IsValid(): diag=Unrecognized: /")
		err = golapack.Dtrti2(Upper, '/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrti2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dtrti2(Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrti2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dtrti2(Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrti2", err)

		//        Dtrtrs
		*srnamt = "Dtrtrs"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtrtrs('/', NoTrans, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Dtrtrs(Upper, '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtrtrs(Upper, NoTrans, '/', 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtrtrs(Upper, NoTrans, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dtrtrs(Upper, NoTrans, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dtrtrs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2))
		chkxer2("Dtrtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dtrtrs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtrtrs", err)

		//        Dtrrfs
		*srnamt = "Dtrrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dtrrfs('/', NoTrans, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Dtrrfs(Upper, '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Dtrrfs(Upper, NoTrans, '/', 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dtrrfs(Upper, NoTrans, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dtrrfs(Upper, NoTrans, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dtrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dtrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dtrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtrrfs", err)

		//        Dtrcon
		*srnamt = "Dtrcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Dtrcon('/', Upper, NonUnit, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtrcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtrcon('1', '/', NonUnit, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtrcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtrcon('1', Upper, '/', 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtrcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtrcon('1', Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtrcon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dtrcon('1', Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtrcon", err)

		//        Dlatrs
		*srnamt = "Dlatrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dlatrs('/', NoTrans, NonUnit, 'N', 0, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		_, err = golapack.Dlatrs(Upper, '/', NonUnit, 'N', 0, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)
		*errt = fmt.Errorf("!diag.IsValid(): diag=Unrecognized: /")
		_, err = golapack.Dlatrs(Upper, NoTrans, '/', 'N', 0, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Dlatrs(Upper, NoTrans, NonUnit, '/', 0, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dlatrs(Upper, NoTrans, NonUnit, 'N', -1, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dlatrs(Upper, NoTrans, NonUnit, 'N', 2, a.Off(0, 0).UpdateRows(1), w, scale, w)
		chkxer2("Dlatrs", err)

	} else if c2 == "tp" {
		//        Test error exits for the packed triangular routines.
		//
		//        Dtptri
		*srnamt = "Dtptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtptri('/', NonUnit, 0, ap)
		chkxer2("Dtptri", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtptri(Upper, '/', 0, ap)
		chkxer2("Dtptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtptri(Upper, NonUnit, -1, ap)
		chkxer2("Dtptri", err)

		//        Dtptrs
		*srnamt = "Dtptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtptrs('/', NoTrans, NonUnit, 0, 0, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Dtptrs(Upper, '/', NonUnit, 0, 0, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtptrs(Upper, NoTrans, '/', 0, 0, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtptrs(Upper, NoTrans, NonUnit, -1, 0, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dtptrs(Upper, NoTrans, NonUnit, 0, -1, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dtptrs(Upper, NoTrans, NonUnit, 2, 1, ap, x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtptrs", err)

		//        Dtprfs
		*srnamt = "Dtprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dtprfs('/', NoTrans, NonUnit, 0, 0, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Dtprfs(Upper, '/', NonUnit, 0, 0, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Dtprfs(Upper, NoTrans, '/', 0, 0, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dtprfs(Upper, NoTrans, NonUnit, -1, 0, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dtprfs(Upper, NoTrans, NonUnit, 0, -1, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dtprfs(Upper, NoTrans, NonUnit, 2, 1, ap, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dtprfs(Upper, NoTrans, NonUnit, 2, 1, ap, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtprfs", err)

		//        Dtpcon
		*srnamt = "Dtpcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Dtpcon('/', Upper, NonUnit, 0, ap, w, &iw)
		chkxer2("Dtpcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtpcon('1', '/', NonUnit, 0, ap, w, &iw)
		chkxer2("Dtpcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtpcon('1', Upper, '/', 0, ap, w, &iw)
		chkxer2("Dtpcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtpcon('1', Upper, NonUnit, -1, ap, w, &iw)
		chkxer2("Dtpcon", err)

		//        Dlatps
		*srnamt = "Dlatps"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dlatps('/', NoTrans, NonUnit, 'N', 0, ap, w, w)
		chkxer2("Dlatps", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Dlatps(Upper, '/', NonUnit, 'N', 0, ap, w, w)
		chkxer2("Dlatps", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dlatps(Upper, NoTrans, '/', 'N', 0, ap, w, w)
		chkxer2("Dlatps", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Dlatps(Upper, NoTrans, NonUnit, '/', 0, ap, w, w)
		chkxer2("Dlatps", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dlatps(Upper, NoTrans, NonUnit, 'N', -1, ap, w, w)
		chkxer2("Dlatps", err)

	} else if c2 == "tb" {
		//        Test error exits for the banded triangular routines.
		//
		//        Dtbtrs
		*srnamt = "Dtbtrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtbtrs('/', NoTrans, NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Dtbtrs(Upper, '/', NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtbtrs(Upper, NoTrans, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtbtrs(Upper, NoTrans, NonUnit, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dtbtrs(Upper, NoTrans, NonUnit, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dtbtrs(Upper, NoTrans, NonUnit, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dtbtrs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2))
		chkxer2("Dtbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dtbtrs(Upper, NoTrans, NonUnit, 2, 0, 1, a.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1))
		chkxer2("Dtbtrs", err)

		//        Dtbrfs
		*srnamt = "Dtbrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dtbrfs('/', NoTrans, NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Dtbrfs(Upper, '/', NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Dtbrfs(Upper, NoTrans, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, 0, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dtbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dtbrfs", err)

		//        Dtbcon
		*srnamt = "Dtbcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Dtbcon('/', Upper, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dtbcon('1', '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Dtbcon('1', Upper, '/', 0, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dtbcon('1', Upper, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dtbcon('1', Upper, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dtbcon('1', Upper, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), w, &iw)
		chkxer2("Dtbcon", err)

		//        Dlatbs
		*srnamt = "Dlatbs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dlatbs('/', NoTrans, NonUnit, 'N', 0, 0, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		_, err = golapack.Dlatbs(Upper, '/', NonUnit, 'N', 0, 0, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("!diag.IsValid(): diag=Unrecognized: /")
		_, err = golapack.Dlatbs(Upper, NoTrans, '/', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Dlatbs(Upper, NoTrans, NonUnit, '/', 0, 0, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dlatbs(Upper, NoTrans, NonUnit, 'N', -1, 0, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dlatbs(Upper, NoTrans, NonUnit, 'N', 1, -1, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dlatbs(Upper, NoTrans, NonUnit, 'N', 2, 1, a.Off(0, 0).UpdateRows(1), w, w)
		chkxer2("Dlatbs", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
