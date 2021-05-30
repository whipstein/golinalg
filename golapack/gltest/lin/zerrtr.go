package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrtr tests the error exits for the COMPLEX*16 triangular routines.
func Zerrtr(path []byte, t *testing.T) {
	var rcond, scale float64
	var info int

	b := cvf(2)
	w := cvf(2)
	x := cvf(2)
	r1 := vf(2)
	r2 := vf(2)
	rw := vf(2)
	a := cmf(2, 2, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]
	a.Set(0, 0, 1.)
	a.Set(0, 1, 2.)
	a.Set(1, 1, 3.)
	a.Set(1, 0, 4.)
	(*ok) = true

	//     Test error exits for the general triangular routines.
	if string(c2) == "TR" {
		//        ZTRTRI
		*srnamt = "ZTRTRI"
		*infot = 1
		golapack.Ztrtri('/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztrtri('U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztrtri('U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRI", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztrtri('U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRI", &info, lerr, ok, t)

		//        ZTRTI2
		*srnamt = "ZTRTI2"
		*infot = 1
		golapack.Ztrti2('/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTI2", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztrti2('U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTI2", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztrti2('U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTI2", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztrti2('U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTI2", &info, lerr, ok, t)

		//        ZTRTRS
		*srnamt = "ZTRTRS"
		*infot = 1
		golapack.Ztrtrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztrtrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztrtrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztrtrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztrtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTRTRS", &info, lerr, ok, t)
		*infot = 7

		//        ZTRRFS
		*srnamt = "ZTRRFS"
		*infot = 1
		golapack.Ztrrfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztrrfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztrrfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztrrfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztrrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Ztrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)
		*infot = 11
		golapack.Ztrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTRRFS", &info, lerr, ok, t)

		//        ZTRCON
		*srnamt = "ZTRCON"
		*infot = 1
		golapack.Ztrcon('/', 'U', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTRCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztrcon('1', '/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTRCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztrcon('1', 'U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTRCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztrcon('1', 'U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTRCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztrcon('1', 'U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTRCON", &info, lerr, ok, t)

		//        ZLATRS
		*srnamt = "ZLATRS"
		*infot = 1
		golapack.Zlatrs('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zlatrs('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zlatrs('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zlatrs('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zlatrs('U', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zlatrs('U', 'N', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATRS", &info, lerr, ok, t)

		//     Test error exits for the packed triangular routines.
	} else if string(c2) == "TP" {
		//        ZTPTRI
		*srnamt = "ZTPTRI"
		*infot = 1
		golapack.Ztptri('/', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), &info)
		Chkxer("ZTPTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztptri('U', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &info)
		Chkxer("ZTPTRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztptri('U', 'N', toPtr(-1), a.CVector(0, 0), &info)
		Chkxer("ZTPTRI", &info, lerr, ok, t)

		//        ZTPTRS
		*srnamt = "ZTPTRS"
		*infot = 1
		golapack.Ztptrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztptrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztptrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztptrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztptrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztptrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTPTRS", &info, lerr, ok, t)

		//        ZTPRFS
		*srnamt = "ZTPRFS"
		*infot = 1
		golapack.Ztprfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztprfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztprfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztprfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztprfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztprfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztprfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTPRFS", &info, lerr, ok, t)

		//        ZTPCON
		*srnamt = "ZTPCON"
		*infot = 1
		golapack.Ztpcon('/', 'U', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), &rcond, w, rw, &info)
		Chkxer("ZTPCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztpcon('1', '/', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), &rcond, w, rw, &info)
		Chkxer("ZTPCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztpcon('1', 'U', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &rcond, w, rw, &info)
		Chkxer("ZTPCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztpcon('1', 'U', 'N', toPtr(-1), a.CVector(0, 0), &rcond, w, rw, &info)
		Chkxer("ZTPCON", &info, lerr, ok, t)

		//        ZLATPS
		*srnamt = "ZLATPS"
		*infot = 1
		golapack.Zlatps('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, &scale, rw, &info)
		Chkxer("ZLATPS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zlatps('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, &scale, rw, &info)
		Chkxer("ZLATPS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zlatps('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, &scale, rw, &info)
		Chkxer("ZLATPS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zlatps('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, &scale, rw, &info)
		Chkxer("ZLATPS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zlatps('U', 'N', 'N', 'N', toPtr(-1), a.CVector(0, 0), x, &scale, rw, &info)
		Chkxer("ZLATPS", &info, lerr, ok, t)

		//     Test error exits for the banded triangular routines.
	} else if string(c2) == "TB" {
		//        ZTBTRS
		*srnamt = "ZTBTRS"
		*infot = 1
		golapack.Ztbtrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztbtrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztbtrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztbtrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztbtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztbtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztbtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztbtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTBTRS", &info, lerr, ok, t)

		//        ZTBRFS
		*srnamt = "ZTBRFS"
		*infot = 1
		golapack.Ztbrfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztbrfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztbrfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztbrfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztbrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztbrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Ztbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, rw, &info)
		Chkxer("ZTBRFS", &info, lerr, ok, t)

		//        ZTBCON
		*srnamt = "ZTBCON"
		*infot = 1
		golapack.Ztbcon('/', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztbcon('1', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztbcon('1', 'U', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztbcon('1', 'U', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztbcon('1', 'U', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztbcon('1', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, rw, &info)
		Chkxer("ZTBCON", &info, lerr, ok, t)

		//        ZLATBS
		*srnamt = "ZLATBS"
		*infot = 1
		golapack.Zlatbs('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zlatbs('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zlatbs('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zlatbs('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zlatbs('U', 'N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 6
		golapack.Zlatbs('U', 'N', 'N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
		*infot = 8
		golapack.Zlatbs('U', 'N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, &scale, rw, &info)
		Chkxer("ZLATBS", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
