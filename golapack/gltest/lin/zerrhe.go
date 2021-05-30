package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrhe tests the error exits for the COMPLEX*16 routines
// for Hermitian indefinite matrices.
func Zerrhe(path []byte, t *testing.T) {
	var anrm, rcond float64
	var i, info, j, nmax int

	nmax = 4
	b := cvf(4)
	e := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	ip := make([]int, 4)
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
		e.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
		ip[j-1] = j
	}
	anrm = 1.0
	(*ok) = true

	if string(c2) == "HE" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//        ZHETRF
		*srnamt = "ZHETRF"
		(*infot) = 1
		golapack.Zhetrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZHETRF", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRF", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZHETRF", &info, lerr, ok, t)

		//        ZHETF2
		*srnamt = "ZHETF2"
		(*infot) = 1
		golapack.Zhetf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2", &info, lerr, ok, t)

		//        ZHETRI
		*srnamt = "ZHETRI"
		(*infot) = 1
		golapack.Zhetri('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetri('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetri('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI", &info, lerr, ok, t)

		//        ZHETRI2
		*srnamt = "ZHETRI2"
		(*infot) = 1
		golapack.Zhetri2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetri2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetri2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2", &info, lerr, ok, t)

		//        ZHETRI2X
		*srnamt = "ZHETRI2X"
		(*infot) = 1
		golapack.Zhetri2x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2X", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetri2x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2X", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetri2x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI2X", &info, lerr, ok, t)

		//        ZHETRS
		*srnamt = "ZHETRS"
		(*infot) = 1
		golapack.Zhetrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhetrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhetrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZHETRS", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS", &info, lerr, ok, t)

		//        ZHERFS
		*srnamt = "ZHERFS"
		(*infot) = 1
		golapack.Zherfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zherfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zherfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zherfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zherfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zherfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zherfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHERFS", &info, lerr, ok, t)

		//        ZHECON
		*srnamt = "ZHECON"
		(*infot) = 1
		golapack.Zhecon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhecon('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhecon('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zhecon('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZHECON", &info, lerr, ok, t)

	} else if string(c2) == "HR" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with rook
		//        (bounded Bunch-Kaufman) diagonal pivoting method.
		//
		//        ZHETRF_ROOK
		*srnamt = "ZHETRF_ROOK"
		(*infot) = 1
		golapack.Zhetrfrook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrfrook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrfrook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZHETRF_ROOK", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrfrook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRF_ROOK", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrfrook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZHETRF_ROOK", &info, lerr, ok, t)

		//        ZHETF2_ROOK
		*srnamt = "ZHETF2_ROOK"
		(*infot) = 1
		golapack.Zhetf2rook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetf2rook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetf2rook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZHETF2_ROOK", &info, lerr, ok, t)

		//        ZHETRI_ROOK
		*srnamt = "ZHETRI_ROOK"
		(*infot) = 1
		golapack.Zhetrirook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrirook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrirook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZHETRI_ROOK", &info, lerr, ok, t)

		//        ZHETRS_ROOK
		*srnamt = "ZHETRS_ROOK"
		(*infot) = 1
		golapack.Zhetrsrook('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrsrook('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_ROOK", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhetrsrook('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_ROOK", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhetrsrook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZHETRS_ROOK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetrsrook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_ROOK", &info, lerr, ok, t)

		//        ZHECON_ROOK
		*srnamt = "ZHECON_ROOK"
		(*infot) = 1
		golapack.Zheconrook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zheconrook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zheconrook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_ROOK", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zheconrook('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZHECON_ROOK", &info, lerr, ok, t)

	} else if string(c2) == "HK" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//        ZHETRF_RK
		*srnamt = "ZHETRF_RK"
		(*infot) = 1
		golapack.Zhetrfrk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_RK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrfrk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_RK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrfrk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZHETRF_RK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetrfrk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRF_RK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetrfrk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("ZHETRF_RK", &info, lerr, ok, t)

		//        ZHETF2_RK
		*srnamt = "ZHETF2_RK"
		(*infot) = 1
		golapack.Zhetf2rk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZHETF2_RK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetf2rk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZHETF2_RK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetf2rk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZHETF2_RK", &info, lerr, ok, t)

		//        ZHETRI_3
		*srnamt = "ZHETRI_3"
		(*infot) = 1
		golapack.Zhetri3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetri3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetri3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRI_3", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("ZHETRI_3", &info, lerr, ok, t)

		//        ZHETRI_3X
		*srnamt = "ZHETRI_3X"
		(*infot) = 1
		golapack.Zhetri3x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3X", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetri3x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3X", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetri3x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRI_3X", &info, lerr, ok, t)

		//        ZHETRS_3
		*srnamt = "ZHETRS_3"
		(*infot) = 1
		golapack.Zhetrs3('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrs3('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_3", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhetrs3('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_3", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhetrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZHETRS_3", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zhetrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_3", &info, lerr, ok, t)

		//        ZHECON_3
		*srnamt = "ZHECON_3"
		(*infot) = 1
		golapack.Zhecon3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhecon3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_3", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhecon3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHECON_3", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhecon3('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, toPtrf64(-1.0), &rcond, w, &info)
		Chkxer("ZHECON_3", &info, lerr, ok, t)

		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with Aasen's algorithm.
	} else if string(c2) == "HA" {
		//        ZHETRF_AA
		*srnamt = "ZHETRF_AA"
		(*infot) = 1
		golapack.Zhetrfaa('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrfaa('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrfaa('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZHETRF_AA", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrfaa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRF_AA", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrfaa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZHETRF_AA", &info, lerr, ok, t)

		//        ZHETRS_AA
		*srnamt = "ZHETRS_AA"
		(*infot) = 1
		golapack.Zhetrsaa('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrsaa('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhetrsaa('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhetrsaa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhetrsaa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhetrsaa('U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhetrsaa('U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZHETRS_AA", &info, lerr, ok, t)

	} else if string(c2) == "S2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        ZHETRF_AA_2STAGE
		*srnamt = "ZHETRF_AA_2STAGE"
		(*infot) = 1
		golapack.Zhetrfaa2stage('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrfaa2stage('U', toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhetrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 2; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zhetrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhetrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRF_AA_2STAGE", &info, lerr, ok, t)

		//        ZHETRS_AA_2STAGE
		*srnamt = "ZHETRS_AA_2STAGE"
		(*infot) = 1
		golapack.Zhetrsaa2stage('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhetrsaa2stage('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhetrsaa2stage('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhetrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhetrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zhetrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRS_AA_STAGE", &info, lerr, ok, t)

	} else if string(c2) == "HP" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite packed matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//        ZHPTRF
		*srnamt = "ZHPTRF"
		(*infot) = 1
		golapack.Zhptrf('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, &info)
		Chkxer("ZHPTRF", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhptrf('U', toPtr(-1), a.CVector(0, 0), &ip, &info)
		Chkxer("ZHPTRF", &info, lerr, ok, t)

		//        ZHPTRI
		*srnamt = "ZHPTRI"
		(*infot) = 1
		golapack.Zhptri('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, w, &info)
		Chkxer("ZHPTRI", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhptri('U', toPtr(-1), a.CVector(0, 0), &ip, w, &info)
		Chkxer("ZHPTRI", &info, lerr, ok, t)

		//        ZHPTRS
		*srnamt = "ZHPTRS"
		(*infot) = 1
		golapack.Zhptrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPTRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhptrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPTRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhptrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPTRS", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhptrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPTRS", &info, lerr, ok, t)

		//        ZHPRFS
		*srnamt = "ZHPRFS"
		(*infot) = 1
		golapack.Zhprfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHPRFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhprfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHPRFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhprfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHPRFS", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zhprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHPRFS", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZHPRFS", &info, lerr, ok, t)

		//        ZHPCON
		*srnamt = "ZHPCON"
		(*infot) = 1
		golapack.Zhpcon('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHPCON", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhpcon('U', toPtr(-1), a.CVector(0, 0), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZHPCON", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhpcon('U', func() *int { y := 1; return &y }(), a.CVector(0, 0), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZHPCON", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
