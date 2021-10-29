package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrtr tests the error exits for the COMPLEX*16 triangular routines.
func zerrtr(path string, t *testing.T) {
	var err error

	b := cvf(2)
	w := cvf(2)
	x := cvf(2)
	r1 := vf(2)
	r2 := vf(2)
	rw := vf(2)
	a := cmf(2, 2, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]
	a.Set(0, 0, 1.)
	a.Set(0, 1, 2.)
	a.Set(1, 1, 3.)
	a.Set(1, 0, 4.)
	(*ok) = true

	//     Test error exits for the general triangular routines.
	if c2 == "tr" {
		//        Ztrtri
		*srnamt = "Ztrtri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztrtri('/', NonUnit, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrtri", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztrtri(Upper, '/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrtri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztrtri(Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrtri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Ztrtri(Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrtri", err)

		//        Ztrti2
		*srnamt = "Ztrti2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Ztrti2('/', NonUnit, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrti2", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Ztrti2(Upper, '/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrti2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Ztrti2(Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrti2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Ztrti2(Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Ztrti2", err)

		//        Ztrtrs
		*srnamt = "Ztrtrs"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztrtrs('/', NoTrans, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Ztrtrs(Upper, '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztrtrs(Upper, NoTrans, '/', 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztrtrs(Upper, NoTrans, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Ztrtrs(Upper, NoTrans, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Ztrtrs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), x.CMatrix(2, opts))
		chkxer2("Ztrtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Ztrtrs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), x.CMatrix(1, opts))
		chkxer2("Ztrtrs", err)

		//        Ztrrfs
		*srnamt = "Ztrrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Ztrrfs('/', NoTrans, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Ztrrfs(Upper, '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Ztrrfs(Upper, NoTrans, '/', 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Ztrrfs(Upper, NoTrans, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Ztrrfs(Upper, NoTrans, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Ztrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Ztrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Ztrrfs(Upper, NoTrans, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(2), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztrrfs", err)

		//        Ztrcon
		*srnamt = "Ztrcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Ztrcon('/', Upper, NonUnit, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztrcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztrcon('1', '/', NonUnit, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztrcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztrcon('1', Upper, '/', 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztrcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztrcon('1', Upper, NonUnit, -1, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztrcon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Ztrcon('1', Upper, NonUnit, 2, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztrcon", err)

		//        Zlatrs
		*srnamt = "Zlatrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zlatrs('/', NoTrans, NonUnit, 'N', 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Zlatrs(Upper, '/', NonUnit, 'N', 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Zlatrs(Upper, NoTrans, '/', 'N', 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Zlatrs(Upper, NoTrans, NonUnit, '/', 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zlatrs(Upper, NoTrans, NonUnit, 'N', -1, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zlatrs(Upper, NoTrans, NonUnit, 'N', 2, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatrs", err)

		//     Test error exits for the packed triangular routines.
	} else if c2 == "tp" {
		//        Ztptri
		*srnamt = "Ztptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztptri('/', NonUnit, 0, a.CVector(0, 0))
		chkxer2("Ztptri", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztptri(Upper, '/', 0, a.CVector(0, 0))
		chkxer2("Ztptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztptri(Upper, NonUnit, -1, a.CVector(0, 0))
		chkxer2("Ztptri", err)

		//        Ztptrs
		*srnamt = "Ztptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztptrs('/', NoTrans, NonUnit, 0, 0, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Ztptrs(Upper, '/', NonUnit, 0, 0, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztptrs(Upper, NoTrans, '/', 0, 0, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztptrs(Upper, NoTrans, NonUnit, -1, 0, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Ztptrs(Upper, NoTrans, NonUnit, 0, -1, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Ztptrs(Upper, NoTrans, NonUnit, 2, 1, a.CVector(0, 0), x.CMatrix(1, opts))
		chkxer2("Ztptrs", err)

		//        Ztprfs
		*srnamt = "Ztprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Ztprfs('/', NoTrans, NonUnit, 0, 0, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Ztprfs(Upper, '/', NonUnit, 0, 0, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Ztprfs(Upper, NoTrans, '/', 0, 0, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Ztprfs(Upper, NoTrans, NonUnit, -1, 0, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Ztprfs(Upper, NoTrans, NonUnit, 0, -1, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Ztprfs(Upper, NoTrans, NonUnit, 2, 1, a.CVector(0, 0), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Ztprfs(Upper, NoTrans, NonUnit, 2, 1, a.CVector(0, 0), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztprfs", err)

		//        Ztpcon
		*srnamt = "Ztpcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Ztpcon('/', Upper, NonUnit, 0, a.CVector(0, 0), w, rw)
		chkxer2("Ztpcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztpcon('1', '/', NonUnit, 0, a.CVector(0, 0), w, rw)
		chkxer2("Ztpcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztpcon('1', Upper, '/', 0, a.CVector(0, 0), w, rw)
		chkxer2("Ztpcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztpcon('1', Upper, NonUnit, -1, a.CVector(0, 0), w, rw)
		chkxer2("Ztpcon", err)

		//        Zlatps
		*srnamt = "Zlatps"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zlatps('/', NoTrans, NonUnit, 'N', 0, a.CVector(0, 0), x, rw)
		chkxer2("Zlatps", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Zlatps(Upper, '/', NonUnit, 'N', 0, a.CVector(0, 0), x, rw)
		chkxer2("Zlatps", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Zlatps(Upper, NoTrans, '/', 'N', 0, a.CVector(0, 0), x, rw)
		chkxer2("Zlatps", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Zlatps(Upper, NoTrans, NonUnit, '/', 0, a.CVector(0, 0), x, rw)
		chkxer2("Zlatps", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zlatps(Upper, NoTrans, NonUnit, 'N', -1, a.CVector(0, 0), x, rw)
		chkxer2("Zlatps", err)

		//     Test error exits for the banded triangular routines.
	} else if c2 == "tb" {
		//        Ztbtrs
		*srnamt = "Ztbtrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztbtrs('/', NoTrans, NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Ztbtrs(Upper, '/', NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztbtrs(Upper, NoTrans, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztbtrs(Upper, NoTrans, NonUnit, -1, 0, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Ztbtrs(Upper, NoTrans, NonUnit, 0, -1, 0, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Ztbtrs(Upper, NoTrans, NonUnit, 0, 0, -1, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Ztbtrs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(1), x.CMatrix(2, opts))
		chkxer2("Ztbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Ztbtrs(Upper, NoTrans, NonUnit, 2, 0, 1, a.Off(0, 0).UpdateRows(1), x.CMatrix(1, opts))
		chkxer2("Ztbtrs", err)

		//        Ztbrfs
		*srnamt = "Ztbrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Ztbrfs('/', NoTrans, NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		err = golapack.Ztbrfs(Upper, '/', NonUnit, 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		err = golapack.Ztbrfs(Upper, NoTrans, '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, 0, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("else if ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(2), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Ztbrfs(Upper, NoTrans, NonUnit, 2, 1, 1, a.Off(0, 0).UpdateRows(2), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Ztbrfs", err)

		//        Ztbcon
		*srnamt = "Ztbcon"
		*errt = fmt.Errorf("!onenrm && norm != 'I': norm='/'")
		_, err = golapack.Ztbcon('/', Upper, NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Ztbcon('1', '/', NonUnit, 0, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Ztbcon('1', Upper, '/', 0, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Ztbcon('1', Upper, NonUnit, -1, 0, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Ztbcon('1', Upper, NonUnit, 0, -1, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Ztbcon('1', Upper, NonUnit, 2, 1, a.Off(0, 0).UpdateRows(1), w, rw)
		chkxer2("Ztbcon", err)

		//        Zlatbs
		*srnamt = "Zlatbs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zlatbs('/', NoTrans, NonUnit, 'N', 0, 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, err = golapack.Zlatbs(Upper, '/', NonUnit, 'N', 0, 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("!nounit && diag != Unit: diag=Unrecognized: /")
		_, err = golapack.Zlatbs(Upper, NoTrans, '/', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("normin != 'Y' && normin != 'N': normin='/'")
		_, err = golapack.Zlatbs(Upper, NoTrans, NonUnit, '/', 0, 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zlatbs(Upper, NoTrans, NonUnit, 'N', -1, 0, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Zlatbs(Upper, NoTrans, NonUnit, 'N', 1, -1, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Zlatbs(Upper, NoTrans, NonUnit, 'N', 2, 1, a.Off(0, 0).UpdateRows(1), x, rw)
		chkxer2("Zlatbs", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
