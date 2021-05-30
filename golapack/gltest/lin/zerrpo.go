package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrpo tests the error exits for the COMPLEX*16 routines
// for Hermitian positive definite matrices.
func Zerrpo(path []byte, t *testing.T) {
	var anrm, rcond float64
	var i, info, j, nmax int

	nmax = 4
	b := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	a := cmf(4, 4, opts)
	af := cmf(4, 4, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
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
	}
	anrm = 1.
	(*ok) = true

	//     Test error exits of the routines that use the Cholesky
	//     decomposition of a Hermitian positive definite matrix.
	if string(c2) == "PO" {
		//        ZPOTRF
		*srnamt = "ZPOTRF"
		*infot = 1
		golapack.Zpotrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpotrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpotrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRF", &info, lerr, ok, t)

		//        ZPOTF2
		*srnamt = "ZPOTF2"
		*infot = 1
		golapack.Zpotf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpotf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpotf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTF2", &info, lerr, ok, t)

		//        ZPOTRI
		*srnamt = "ZPOTRI"
		*infot = 1
		golapack.Zpotri('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpotri('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRI", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpotri('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRI", &info, lerr, ok, t)

		//        ZPOTRS
		*srnamt = "ZPOTRS"
		*infot = 1
		golapack.Zpotrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpotrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpotrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpotrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZPOTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zpotrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOTRS", &info, lerr, ok, t)

		//        ZPORFS
		*srnamt = "ZPORFS"
		*infot = 1
		golapack.Zporfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zporfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zporfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Zporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)
		*infot = 11
		golapack.Zporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPORFS", &info, lerr, ok, t)

		//        ZPOCON
		*srnamt = "ZPOCON"
		*infot = 1
		golapack.Zpocon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPOCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpocon('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPOCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpocon('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPOCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpocon('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), toPtrf64(-anrm), &rcond, w, r, &info)
		Chkxer("ZPOCON", &info, lerr, ok, t)

		//        ZPOEQU
		*srnamt = "ZPOEQU"
		*infot = 1
		golapack.Zpoequ(toPtr(-1), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPOEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpoequ(func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPOEQU", &info, lerr, ok, t)

		//     Test error exits of the routines that use the Cholesky
		//     decomposition of a Hermitian positive definite packed matrix.
	} else if string(c2) == "PP" {
		//        ZPPTRF
		*srnamt = "ZPPTRF"
		*infot = 1
		golapack.Zpptrf('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &info)
		Chkxer("ZPPTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpptrf('U', toPtr(-1), a.CVector(0, 0), &info)
		Chkxer("ZPPTRF", &info, lerr, ok, t)

		//        ZPPTRI
		*srnamt = "ZPPTRI"
		*infot = 1
		golapack.Zpptri('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &info)
		Chkxer("ZPPTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpptri('U', toPtr(-1), a.CVector(0, 0), &info)
		Chkxer("ZPPTRI", &info, lerr, ok, t)

		//        ZPPTRS
		*srnamt = "ZPPTRS"
		*infot = 1
		golapack.Zpptrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpptrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpptrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpptrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPTRS", &info, lerr, ok, t)

		//        ZPPRFS
		*srnamt = "ZPPRFS"
		*infot = 1
		golapack.Zpprfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPPRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpprfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPPRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpprfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPPRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zpprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPPRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Zpprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPPRFS", &info, lerr, ok, t)

		//        ZPPCON
		*srnamt = "ZPPCON"
		*infot = 1
		golapack.Zppcon('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &anrm, &rcond, w, r, &info)
		Chkxer("ZPPCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Zppcon('U', toPtr(-1), a.CVector(0, 0), &anrm, &rcond, w, r, &info)
		Chkxer("ZPPCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Zppcon('U', func() *int { y := 1; return &y }(), a.CVector(0, 0), toPtrf64(-anrm), &rcond, w, r, &info)
		Chkxer("ZPPCON", &info, lerr, ok, t)

		//        ZPPEQU
		*srnamt = "ZPPEQU"
		*infot = 1
		golapack.Zppequ('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), r1, &rcond, &anrm, &info)
		Chkxer("ZPPEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Zppequ('U', toPtr(-1), a.CVector(0, 0), r1, &rcond, &anrm, &info)
		Chkxer("ZPPEQU", &info, lerr, ok, t)

		//     Test error exits of the routines that use the Cholesky
		//     decomposition of a Hermitian positive definite band matrix.
	} else if string(c2) == "PB" {
		//        ZPBTRF
		*srnamt = "ZPBTRF"
		*infot = 1
		golapack.Zpbtrf('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbtrf('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbtrf('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRF", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpbtrf('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRF", &info, lerr, ok, t)

		//        ZPBTF2
		*srnamt = "ZPBTF2"
		*infot = 1
		golapack.Zpbtf2('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbtf2('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTF2", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbtf2('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTF2", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpbtf2('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTF2", &info, lerr, ok, t)

		//        ZPBTRS
		*srnamt = "ZPBTRS"
		*infot = 1
		golapack.Zpbtrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbtrs('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbtrs('U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpbtrs('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpbtrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Zpbtrs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBTRS", &info, lerr, ok, t)

		//        ZPBRFS
		*srnamt = "ZPBRFS"
		*infot = 1
		golapack.Zpbrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbrfs('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbrfs('U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpbrfs('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Zpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Zpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Zpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZPBRFS", &info, lerr, ok, t)

		//        ZPBCON
		*srnamt = "ZPBCON"
		*infot = 1
		golapack.Zpbcon('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbcon('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbcon('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPBCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpbcon('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZPBCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpbcon('U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), toPtrf64(-anrm), &rcond, w, r, &info)
		Chkxer("ZPBCON", &info, lerr, ok, t)

		//        ZPBEQU
		*srnamt = "ZPBEQU"
		*infot = 1
		golapack.Zpbequ('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPBEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbequ('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPBEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbequ('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPBEQU", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpbequ('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("ZPBEQU", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
