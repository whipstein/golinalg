package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

//golapack.Zerrsy tests the error exits for the COMPLEX*16 routines
// for symmetric indefinite matrices.
func Zerrsy(path []byte, t *testing.T) {
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

	if string(c2) == "SY" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//       golapack.ZSYTRF
		*srnamt = "ZSYTRF"
		(*infot) = 1
		golapack.Zsytrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZSYTRF", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRF", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZSYTRF", &info, lerr, ok, t)

		//       golapack.ZSYTF2
		*srnamt = "ZSYTF2"
		(*infot) = 1
		golapack.Zsytf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2", &info, lerr, ok, t)

		//       golapack.ZSYTRI
		*srnamt = "ZSYTRI"
		(*infot) = 1
		golapack.Zsytri('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytri('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytri('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI", &info, lerr, ok, t)

		//       golapack.ZSYTRI2
		*srnamt = "ZSYTRI2"
		(*infot) = 1
		golapack.Zsytri2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytri2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytri2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2", &info, lerr, ok, t)

		//       golapack.ZSYTRI2X
		*srnamt = "ZSYTRI2X"
		(*infot) = 1
		golapack.Zsytri2x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2X", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytri2x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2X", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytri2x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI2X", &info, lerr, ok, t)

		//       golapack.ZSYTRS
		*srnamt = "ZSYTRS"
		(*infot) = 1
		golapack.Zsytrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsytrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsytrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZSYTRS", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS", &info, lerr, ok, t)

		//       golapack.ZSYRFS
		*srnamt = "ZSYRFS"
		(*infot) = 1
		golapack.Zsyrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsyrfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsyrfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSYRFS", &info, lerr, ok, t)

		//       golapack.ZSYCON
		*srnamt = "ZSYCON"
		(*infot) = 1
		golapack.Zsycon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsycon('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsycon('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zsycon('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZSYCON", &info, lerr, ok, t)

	} else if string(c2) == "SR" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) diagonal pivoting method.
		//
		//       golapack.ZSYTRF_ROOK
		*srnamt = "ZSYTRF_ROOK"
		(*infot) = 1
		golapack.Zsytrfrook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrfrook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrfrook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZSYTRF_ROOK", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrfrook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRF_ROOK", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrfrook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZSYTRF_ROOK", &info, lerr, ok, t)

		//       golapack.ZSYTF2_ROOK
		*srnamt = "ZSYTF2_ROOK"
		(*infot) = 1
		golapack.Zsytf2rook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytf2rook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytf2rook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZSYTF2_ROOK", &info, lerr, ok, t)

		//       golapack.ZSYTRI_ROOK
		*srnamt = "ZSYTRI_ROOK"
		(*infot) = 1
		golapack.Zsytrirook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrirook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrirook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("ZSYTRI_ROOK", &info, lerr, ok, t)

		//       golapack.ZSYTRS_ROOK
		*srnamt = "ZSYTRS_ROOK"
		(*infot) = 1
		golapack.Zsytrsrook('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrsrook('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_ROOK", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsytrsrook('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_ROOK", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsytrsrook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZSYTRS_ROOK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytrsrook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_ROOK", &info, lerr, ok, t)

		//       golapack.ZSYCON_ROOK
		*srnamt = "ZSYCON_ROOK"
		(*infot) = 1
		golapack.Zsyconrook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_ROOK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsyconrook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_ROOK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsyconrook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_ROOK", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zsyconrook('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZSYCON_ROOK", &info, lerr, ok, t)

	} else if string(c2) == "SK" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//       golapack.ZSYTRF_RK
		*srnamt = "ZSYTRF_RK"
		(*infot) = 1
		golapack.Zsytrfrk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_RK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrfrk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_RK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrfrk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZSYTRF_RK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytrfrk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRF_RK", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytrfrk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("ZSYTRF_RK", &info, lerr, ok, t)

		//       golapack.ZSYTF2_RK
		*srnamt = "ZSYTF2_RK"
		(*infot) = 1
		golapack.Zsytf2rk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZSYTF2_RK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytf2rk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZSYTF2_RK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytf2rk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("ZSYTF2_RK", &info, lerr, ok, t)

		//       golapack.ZSYTRI_3
		*srnamt = "ZSYTRI_3"
		(*infot) = 1
		golapack.Zsytri3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytri3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytri3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRI_3", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("ZSYTRI_3", &info, lerr, ok, t)

		//       golapack.ZSYTRI_3X
		*srnamt = "ZSYTRI_3X"
		(*infot) = 1
		golapack.Zsytri3x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3X", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytri3x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3X", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytri3x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRI_3X", &info, lerr, ok, t)

		//       golapack.ZSYTRS_3
		*srnamt = "ZSYTRS_3"
		(*infot) = 1
		golapack.Zsytrs3('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrs3('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_3", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsytrs3('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_3", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsytrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZSYTRS_3", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zsytrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_3", &info, lerr, ok, t)

		//       golapack.ZSYCON_3
		*srnamt = "ZSYCON_3"
		(*infot) = 1
		golapack.Zsycon3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_3", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsycon3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_3", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsycon3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSYCON_3", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsycon3('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, toPtrf64(-1.0), &rcond, w, &info)
		Chkxer("ZSYCON_3", &info, lerr, ok, t)

	} else if string(c2) == "SP" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite packed matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//       golapack.ZSPTRF
		*srnamt = "ZSPTRF"
		(*infot) = 1
		golapack.Zsptrf('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, &info)
		Chkxer("ZSPTRF", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsptrf('U', toPtr(-1), a.CVector(0, 0), &ip, &info)
		Chkxer("ZSPTRF", &info, lerr, ok, t)

		//       golapack.ZSPTRI
		*srnamt = "ZSPTRI"
		(*infot) = 1
		golapack.Zsptri('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, w, &info)
		Chkxer("ZSPTRI", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsptri('U', toPtr(-1), a.CVector(0, 0), &ip, w, &info)
		Chkxer("ZSPTRI", &info, lerr, ok, t)

		//       golapack.ZSPTRS
		*srnamt = "ZSPTRS"
		(*infot) = 1
		golapack.Zsptrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPTRS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsptrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPTRS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsptrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPTRS", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsptrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPTRS", &info, lerr, ok, t)

		//       golapack.ZSPRFS
		*srnamt = "ZSPRFS"
		(*infot) = 1
		golapack.Zsprfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSPRFS", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsprfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSPRFS", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsprfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSPRFS", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSPRFS", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zsprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZSPRFS", &info, lerr, ok, t)

		//       golapack.ZSPCON
		*srnamt = "ZSPCON"
		(*infot) = 1
		golapack.Zspcon('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSPCON", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zspcon('U', toPtr(-1), a.CVector(0, 0), &ip, &anrm, &rcond, w, &info)
		Chkxer("ZSPCON", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zspcon('U', func() *int { y := 1; return &y }(), a.CVector(0, 0), &ip, toPtrf64(-anrm), &rcond, w, &info)
		Chkxer("ZSPCON", &info, lerr, ok, t)

	} else if string(c2) == "SA" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//       golapack.ZSYTRF_AA
		*srnamt = "ZSYTRF_AA"
		(*infot) = 1
		golapack.Zsytrfaa('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrfaa('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrfaa('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("ZSYTRF_AA", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrfaa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRF_AA", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrfaa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("ZSYTRF_AA", &info, lerr, ok, t)

		//       golapack.ZSYTRS_AA
		*srnamt = "ZSYTRS_AA"
		(*infot) = 1
		golapack.Zsytrsaa('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrsaa('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsytrsaa('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsytrsaa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zsytrsaa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA", &info, lerr, ok, t)

	} else if string(c2) == "S2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//       golapack.ZSYTRF_AA_2STAGE
		*srnamt = "ZSYTRF_AA_2STAGE"
		(*infot) = 1
		golapack.Zsytrfaa2stage('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrfaa2stage('U', toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zsytrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 2; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zsytrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRF_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zsytrfaa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYTRF_AA_2STAGE", &info, lerr, ok, t)

		//        CHETRS_AA_2STAGE
		*srnamt = "ZSYTRS_AA_2STAGE"
		(*infot) = 1
		golapack.Zsytrsaa2stage('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zsytrsaa2stage('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zsytrsaa2stage('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zsytrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zsytrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_2STAGE", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zsytrsaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYTRS_AA_STAGE", &info, lerr, ok, t)

	}

	//     Print a summary line.
	Alaesm(path, ok)
}
