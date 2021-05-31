package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrtr tests the error exits for the DOUBLE PRECISION triangular
// routines.
func Derrtr(path []byte, t *testing.T) {
	var rcond, scale float64
	var info int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	// nmax = 2

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

	if string(c2) == "TR" {
		//        Test error exits for the general triangular routines.
		//
		//        DTRTRI
		*srnamt = "DTRTRI"
		*infot = 1
		golapack.Dtrtri('/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrtri('U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtrtri('U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRI", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtrtri('U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRI", &info, lerr, ok, t)

		//        DTRTI2
		*srnamt = "DTRTI2"
		*infot = 1
		golapack.Dtrti2('/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTI2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrti2('U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTI2", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtrti2('U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTI2", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtrti2('U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTI2", &info, lerr, ok, t)

		//        DTRTRS
		*srnamt = "DTRTRS"
		*infot = 1
		golapack.Dtrtrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrtrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtrtrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtrtrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtrtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtrtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)
		*infot = 9
		golapack.Dtrtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTRTRS", &info, lerr, ok, t)

		//        DTRRFS
		*srnamt = "DTRRFS"
		*infot = 1
		golapack.Dtrrfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrrfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtrrfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtrrfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtrrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Dtrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)
		*infot = 11
		golapack.Dtrrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTRRFS", &info, lerr, ok, t)

		//        DTRCON
		*srnamt = "DTRCON"
		*infot = 1
		golapack.Dtrcon('/', 'U', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTRCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrcon('1', '/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTRCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtrcon('1', 'U', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTRCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtrcon('1', 'U', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTRCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtrcon('1', 'U', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTRCON", &info, lerr, ok, t)

		//        DLATRS
		*srnamt = "DLATRS"
		*infot = 1
		golapack.Dlatrs('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dlatrs('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dlatrs('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dlatrs('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dlatrs('U', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dlatrs('U', 'N', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATRS", &info, lerr, ok, t)

	} else if string(c2) == "TP" {
		//        Test error exits for the packed triangular routines.
		//
		//        DTPTRI
		*srnamt = "DTPTRI"
		*infot = 1
		golapack.Dtptri('/', 'N', func() *int { y := 0; return &y }(), ap, &info)
		Chkxer("DTPTRI", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtptri('U', '/', func() *int { y := 0; return &y }(), ap, &info)
		Chkxer("DTPTRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtptri('U', 'N', toPtr(-1), ap, &info)
		Chkxer("DTPTRI", &info, lerr, ok, t)

		//        DTPTRS
		*srnamt = "DTPTRS"
		*infot = 1
		golapack.Dtptrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtptrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtptrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtptrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtptrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtptrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTPTRS", &info, lerr, ok, t)

		//        DTPRFS
		*srnamt = "DTPRFS"
		*infot = 1
		golapack.Dtprfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtprfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtprfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtprfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtprfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtprfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtprfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), ap, b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTPRFS", &info, lerr, ok, t)

		//        DTPCON
		*srnamt = "DTPCON"
		*infot = 1
		golapack.Dtpcon('/', 'U', 'N', func() *int { y := 0; return &y }(), ap, &rcond, w, &iw, &info)
		Chkxer("DTPCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtpcon('1', '/', 'N', func() *int { y := 0; return &y }(), ap, &rcond, w, &iw, &info)
		Chkxer("DTPCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtpcon('1', 'U', '/', func() *int { y := 0; return &y }(), ap, &rcond, w, &iw, &info)
		Chkxer("DTPCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtpcon('1', 'U', 'N', toPtr(-1), ap, &rcond, w, &iw, &info)
		Chkxer("DTPCON", &info, lerr, ok, t)

		//        DLATPS
		*srnamt = "DLATPS"
		*infot = 1
		golapack.Dlatps('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), ap, w, &scale, w, &info)
		Chkxer("DLATPS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dlatps('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), ap, w, &scale, w, &info)
		Chkxer("DLATPS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dlatps('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), ap, w, &scale, w, &info)
		Chkxer("DLATPS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dlatps('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), ap, w, &scale, w, &info)
		Chkxer("DLATPS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dlatps('U', 'N', 'N', 'N', toPtr(-1), ap, w, &scale, w, &info)
		Chkxer("DLATPS", &info, lerr, ok, t)

	} else if string(c2) == "TB" {
		//        Test error exits for the banded triangular routines.
		//
		//        DTBTRS
		*srnamt = "DTBTRS"
		*infot = 1
		golapack.Dtbtrs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtbtrs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtbtrs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtbtrs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtbtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtbtrs('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtbtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtbtrs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTBTRS", &info, lerr, ok, t)

		//        DTBRFS
		*srnamt = "DTBRFS"
		*infot = 1
		golapack.Dtbrfs('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtbrfs('U', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtbrfs('U', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtbrfs('U', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtbrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtbrfs('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), x, func() *int { y := 2; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Dtbrfs('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), x, func() *int { y := 1; return &y }(), r1, r2, w, &iw, &info)
		Chkxer("DTBRFS", &info, lerr, ok, t)

		//        DTBCON
		*srnamt = "DTBCON"
		*infot = 1
		golapack.Dtbcon('/', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtbcon('1', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtbcon('1', 'U', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtbcon('1', 'U', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtbcon('1', 'U', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtbcon('1', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &rcond, w, &iw, &info)
		Chkxer("DTBCON", &info, lerr, ok, t)

		//        DLATBS
		*srnamt = "DLATBS"
		*infot = 1
		golapack.Dlatbs('/', 'N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dlatbs('U', '/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dlatbs('U', 'N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dlatbs('U', 'N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dlatbs('U', 'N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 6
		golapack.Dlatbs('U', 'N', 'N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dlatbs('U', 'N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &scale, w, &info)
		Chkxer("DLATBS", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
