package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrpo tests the error exits for the DOUBLE PRECISION routines
// for symmetric positive definite matrices.
func Derrpo(path []byte, t *testing.T) {
	var anrm, rcond float64
	var i, info, j, nmax int
	iw := make([]int, 4)
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
	w := vf(3 * nmax)
	x := mf(nmax, 1, opts)
	r1 := vf(4)
	r2 := vf(4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.SetIdx(j-1, 0.)
		iw[j-1] = j
	}
	(*ok) = true

	if string(c2) == "PO" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite matrix.
		//
		//        DPOTRF
		*srnamt = "DPOTRF"
		*infot = 1
		golapack.Dpotrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpotrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpotrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRF", &info, lerr, ok, t)

		//        DPOTF2
		*srnamt = "DPOTF2"
		*infot = 1
		golapack.Dpotf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpotf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpotf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTF2", &info, lerr, ok, t)

		//        DPOTRI
		*srnamt = "DPOTRI"
		*infot = 1
		golapack.Dpotri('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpotri('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRI", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpotri('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRI", &info, lerr, ok, t)

		//        DPOTRS
		*srnamt = "DPOTRS"
		*infot = 1
		golapack.Dpotrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpotrs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpotrs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpotrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), &info)
		Chkxer("DPOTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dpotrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPOTRS", &info, lerr, ok, t)

		//        DPORFS
		*srnamt = "DPORFS"
		*infot = 1
		golapack.Dporfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dporfs('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dporfs('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Dporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)
		*infot = 11
		golapack.Dporfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPORFS", &info, lerr, ok, t)

		//        DPOCON
		*srnamt = "DPOCON"
		*infot = 1
		golapack.Dpocon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPOCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpocon('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPOCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpocon('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPOCON", &info, lerr, ok, t)

		//        DPOEQU
		*srnamt = "DPOEQU"
		*infot = 1
		golapack.Dpoequ(toPtr(-1), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPOEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpoequ(func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPOEQU", &info, lerr, ok, t)

	} else if string(c2) == "PP" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite packed matrix.
		//
		//        DPPTRF
		*srnamt = "DPPTRF"
		*infot = 1
		golapack.Dpptrf('/', func() *int { y := 0; return &y }(), ap, &info)
		Chkxer("DPPTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpptrf('U', toPtr(-1), ap, &info)
		Chkxer("DPPTRF", &info, lerr, ok, t)

		//        DPPTRI
		*srnamt = "DPPTRI"
		*infot = 1
		golapack.Dpptri('/', func() *int { y := 0; return &y }(), ap, &info)
		Chkxer("DPPTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpptri('U', toPtr(-1), ap, &info)
		Chkxer("DPPTRI", &info, lerr, ok, t)

		//        DPPTRS
		*srnamt = "DPPTRS"
		*infot = 1
		golapack.Dpptrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPPTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpptrs('U', toPtr(-1), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPPTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpptrs('U', func() *int { y := 0; return &y }(), toPtr(-1), ap, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPPTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpptrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPPTRS", &info, lerr, ok, t)

		//        DPPRFS
		*srnamt = "DPPRFS"
		*infot = 1
		golapack.Dpprfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, afp, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPPRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpprfs('U', toPtr(-1), func() *int { y := 0; return &y }(), ap, afp, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPPRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpprfs('U', func() *int { y := 0; return &y }(), toPtr(-1), ap, afp, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPPRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dpprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, afp, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPPRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Dpprfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, afp, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPPRFS", &info, lerr, ok, t)

		//        DPPCON
		*srnamt = "DPPCON"
		*infot = 1
		golapack.Dppcon('/', func() *int { y := 0; return &y }(), ap, &anrm, &rcond, w, &iw, &info)
		Chkxer("DPPCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dppcon('U', toPtr(-1), ap, &anrm, &rcond, w, &iw, &info)
		Chkxer("DPPCON", &info, lerr, ok, t)

		//        DPPEQU
		*srnamt = "DPPEQU"
		*infot = 1
		golapack.Dppequ('/', func() *int { y := 0; return &y }(), ap, r1, &rcond, &anrm, &info)
		Chkxer("DPPEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Dppequ('U', toPtr(-1), ap, r1, &rcond, &anrm, &info)
		Chkxer("DPPEQU", &info, lerr, ok, t)

	} else if string(c2) == "PB" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite band matrix.
		//
		//        DPBTRF
		*srnamt = "DPBTRF"
		*infot = 1
		golapack.Dpbtrf('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbtrf('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbtrf('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRF", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpbtrf('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRF", &info, lerr, ok, t)

		//        DPBTF2
		*srnamt = "DPBTF2"
		*infot = 1
		golapack.Dpbtf2('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbtf2('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTF2", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbtf2('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTF2", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpbtf2('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTF2", &info, lerr, ok, t)

		//        DPBTRS
		*srnamt = "DPBTRS"
		*infot = 1
		golapack.Dpbtrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbtrs('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbtrs('U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpbtrs('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpbtrs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dpbtrs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &info)
		Chkxer("DPBTRS", &info, lerr, ok, t)

		//        DPBRFS
		*srnamt = "DPBRFS"
		*infot = 1
		golapack.Dpbrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbrfs('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbrfs('U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpbrfs('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Dpbrfs('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DPBRFS", &info, lerr, ok, t)

		//        DPBCON
		*srnamt = "DPBCON"
		*infot = 1
		golapack.Dpbcon('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbcon('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbcon('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPBCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpbcon('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, &iw, &info)
		Chkxer("DPBCON", &info, lerr, ok, t)

		//        DPBEQU
		*srnamt = "DPBEQU"
		*infot = 1
		golapack.Dpbequ('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPBEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbequ('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPBEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbequ('U', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPBEQU", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpbequ('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), r1, &rcond, &anrm, &info)
		Chkxer("DPBEQU", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
