package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrsy tests the error exits for the DOUBLE PRECISION routines
// for symmetric indefinite matrices.
func Derrsy(path []byte, t *testing.T) {
	var anrm, rcond float64
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4

	a := mf(4, 4, opts)
	af := mf(4, 4, opts)
	ap := vf(4 * 4)
	afp := vf(4 * 4)
	b := mf(4, 1, opts)
	e := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	w := vf(3 * nmax)
	x := mf(4, 1, opts)
	ip := make([]int, 4)
	iw := make([]int, 4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		e.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.SetIdx(j-1, 0.)
		ip[j-1] = j
		iw[j-1] = j
	}
	anrm = 1.0
	rcond = 1.0
	(*ok) = true

	if string(c2) == "SY" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//        DSYTRF
		*srnamt = "DSYTRF"
		*infot = 1
		golapack.Dsytrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("DSYTRF", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsytrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRF", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsytrf('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("DSYTRF", &info, lerr, ok, t)

		//        DSYTF2
		*srnamt = "DSYTF2"
		*infot = 1
		golapack.Dsytf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2", &info, lerr, ok, t)

		//        DSYTRI
		*srnamt = "DSYTRI"
		*infot = 1
		golapack.Dsytri('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytri('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytri('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI", &info, lerr, ok, t)

		//        DSYTRI2
		*srnamt = "DSYTRI2"
		*infot = 1
		golapack.Dsytri2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, &(iw[0]), &info)
		Chkxer("DSYTRI2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytri2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, x, &(iw[0]), &info)
		Chkxer("DSYTRI2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytri2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, &(iw[0]), &info)
		Chkxer("DSYTRI2", &info, lerr, ok, t)

		//        DSYTRI2X
		*srnamt = "DSYTRI2X"
		*infot = 1
		golapack.Dsytri2x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI2X", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytri2x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI2X", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytri2x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI2X", &info, lerr, ok, t)

		//        DSYTRS
		*srnamt = "DSYTRS"
		*infot = 1
		golapack.Dsytrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsytrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsytrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 2; return &y }(), &info)
		Chkxer("DSYTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsytrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS", &info, lerr, ok, t)

		//        DSYRFS
		*srnamt = "DSYRFS"
		*infot = 1
		golapack.Dsyrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyrfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyrfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsyrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSYRFS", &info, lerr, ok, t)

		//        DSYCON
		*srnamt = "DSYCON"
		*infot = 1
		golapack.Dsycon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsycon('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsycon('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsycon('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, func() *float64 { y := -1.0; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DSYCON", &info, lerr, ok, t)

	} else if string(c2) == "SR" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting.
		//
		//        DSYTRF_ROOK
		*srnamt = "DSYTRF_ROOK"
		*infot = 1
		golapack.DsytrfRook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrfRook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_ROOK", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrfRook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, func() *int { y := 4; return &y }(), &info)
		Chkxer("DSYTRF_ROOK", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrfRook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRF_ROOK", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrfRook('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, x, toPtr(-2), &info)
		Chkxer("DSYTRF_ROOK", &info, lerr, ok, t)

		//        DSYTF2_ROOK
		*srnamt = "DSYTF2_ROOK"
		*infot = 1
		golapack.Dsytf2Rook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytf2Rook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2_ROOK", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytf2Rook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DSYTF2_ROOK", &info, lerr, ok, t)

		//        DSYTRI_ROOK
		*srnamt = "DSYTRI_ROOK"
		*infot = 1
		golapack.DsytriRook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytriRook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI_ROOK", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytriRook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, &info)
		Chkxer("DSYTRI_ROOK", &info, lerr, ok, t)

		//        DSYTRS_ROOK
		*srnamt = "DSYTRS_ROOK"
		*infot = 1
		golapack.DsytrsRook('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrsRook('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_ROOK", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrsRook('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_ROOK", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrsRook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 2; return &y }(), &info)
		Chkxer("DSYTRS_ROOK", &info, lerr, ok, t)
		*infot = 8
		golapack.DsytrsRook('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_ROOK", &info, lerr, ok, t)

		//        DSYCON_ROOK
		*srnamt = "DSYCON_ROOK"
		*infot = 1
		golapack.DsyconRook('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsyconRook('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_ROOK", &info, lerr, ok, t)
		*infot = 4
		golapack.DsyconRook('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_ROOK", &info, lerr, ok, t)
		*infot = 6
		golapack.DsyconRook('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, func() *float64 { y := -1.0; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DSYCON_ROOK", &info, lerr, ok, t)

	} else if string(c2) == "SK" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//        DSYTRF_RK
		*srnamt = "DSYTRF_RK"
		*infot = 1
		golapack.DsytrfRk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_RK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrfRk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_RK", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrfRk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_RK", &info, lerr, ok, t)
		*infot = 8
		golapack.DsytrfRk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRF_RK", &info, lerr, ok, t)
		*infot = 8
		golapack.DsytrfRk('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("DSYTRF_RK", &info, lerr, ok, t)

		//        DSYTF2_RK
		*srnamt = "DSYTF2_RK"
		*infot = 1
		golapack.Dsytf2Rk('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("DSYTF2_RK", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytf2Rk('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("DSYTF2_RK", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytf2Rk('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &info)
		Chkxer("DSYTF2_RK", &info, lerr, ok, t)

		//        DSYTRI_3
		*srnamt = "DSYTRI_3"
		*infot = 1
		golapack.Dsytri3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytri3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytri3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsytri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRI_3", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsytri3('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, toPtr(-2), &info)
		Chkxer("DSYTRI_3", &info, lerr, ok, t)

		//        DSYTRI_3X
		*srnamt = "DSYTRI_3X"
		*infot = 1
		golapack.Dsytri3x('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3X", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytri3x('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3X", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytri3x('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRI_3X", &info, lerr, ok, t)

		//        DSYTRS_3
		*srnamt = "DSYTRS_3"
		*infot = 1
		golapack.Dsytrs3('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytrs3('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_3", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsytrs3('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_3", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsytrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b, func() *int { y := 2; return &y }(), &info)
		Chkxer("DSYTRS_3", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsytrs3('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), e, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_3", &info, lerr, ok, t)

		//        DSYCON_3
		*srnamt = "DSYCON_3"
		*infot = 1
		golapack.Dsycon3('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsycon3('U', toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_3", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsycon3('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSYCON_3", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsycon3('U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, func() *float64 { y := -1.0; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DSYCON_3", &info, lerr, ok, t)

	} else if string(c2) == "SA" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        DSYTRF_AA
		*srnamt = "DSYTRF_AA"
		*infot = 1
		golapack.DsytrfAa('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrfAa('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrfAa('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 4; return &y }(), &info)
		Chkxer("DSYTRF_AA", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrfAa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRF_AA", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrfAa('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, toPtr(-2), &info)
		Chkxer("DSYTRF_AA", &info, lerr, ok, t)

		//        DSYTRS_AA
		*srnamt = "DSYTRS_AA"
		*infot = 1
		golapack.DsytrsAa('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrsAa('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrsAa('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrsAa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 8
		golapack.DsytrsAa('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 10
		golapack.DsytrsAa('U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)
		*infot = 10
		golapack.DsytrsAa('U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b, func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("DSYTRS_AA", &info, lerr, ok, t)

	} else if string(c2) == "S2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        DSYTRF_AA_2STAGE
		*srnamt = "DSYTRF_AA_2STAGE"
		*infot = 1
		golapack.DsytrfAa2stage('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrfAa2stage('U', toPtr(-1), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrfAa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.DsytrfAa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRF_AA_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.DsytrfAa2stage('U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), w, func() *int { y := 8; return &y }(), &ip, &ip, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DSYTRF_AA_2STAGE", &info, lerr, ok, t)

		//        DSYTRS_AA_2STAGE
		*srnamt = "DSYTRS_AA_2STAGE"
		*infot = 1
		golapack.DsytrsAa2stage('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrsAa2stage('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrsAa2stage('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrsAa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrsAa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.DsytrsAa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), w, func() *int { y := 8; return &y }(), &ip, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSYTRS_AA_STAGE", &info, lerr, ok, t)

	} else if string(c2) == "SP" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite packed matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//        DSPTRF
		*srnamt = "DSPTRF"
		*infot = 1
		golapack.Dsptrf('/', func() *int { y := 0; return &y }(), ap, &ip, &info)
		Chkxer("DSPTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsptrf('U', toPtr(-1), ap, &ip, &info)
		Chkxer("DSPTRF", &info, lerr, ok, t)

		//        DSPTRI
		*srnamt = "DSPTRI"
		*infot = 1
		golapack.Dsptri('/', func() *int { y := 0; return &y }(), ap, &ip, w, &info)
		Chkxer("DSPTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsptri('U', toPtr(-1), ap, &ip, w, &info)
		Chkxer("DSPTRI", &info, lerr, ok, t)

		//        DSPTRS
		*srnamt = "DSPTRS"
		*infot = 1
		golapack.Dsptrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSPTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsptrs('U', toPtr(-1), func() *int { y := 0; return &y }(), ap, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSPTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsptrs('U', func() *int { y := 0; return &y }(), toPtr(-1), ap, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSPTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsptrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, &ip, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DSPTRS", &info, lerr, ok, t)

		//        DSPRFS
		*srnamt = "DSPRFS"
		*infot = 1
		golapack.Dsprfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, afp, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSPRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsprfs('U', toPtr(-1), func() *int { y := 0; return &y }(), ap, afp, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSPRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsprfs('U', func() *int { y := 0; return &y }(), toPtr(-1), ap, afp, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSPRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, afp, &ip, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSPRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, afp, &ip, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DSPRFS", &info, lerr, ok, t)

		//        DSPCON
		*srnamt = "DSPCON"
		*infot = 1
		golapack.Dspcon('/', func() *int { y := 0; return &y }(), ap, &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSPCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspcon('U', toPtr(-1), ap, &ip, &anrm, &rcond, w, &iw, &info)
		Chkxer("DSPCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Dspcon('U', func() *int { y := 1; return &y }(), ap, &ip, func() *float64 { y := -1.0; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DSPCON", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
