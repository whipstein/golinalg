package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrls tests the error exits for the DOUBLE PRECISION least squares
// driver routines (DGELS, SGELSS, SGELSY, SGELSD).
func Derrls(path []byte, t *testing.T) {
	var rcond float64
	var info, irnk int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt
	ip := make([]int, 2)

	a := mf(2, 2, opts)
	b := mf(2, 2, opts)
	s := vf(2)
	w := vf(2)

	c2 := path[1:3]
	a.Set(0, 0, 1.0)
	a.Set(0, 1, 2.0)
	a.Set(1, 1, 3.0)
	a.Set(1, 0, 4.0)
	(*ok) = true

	if string(c2) == "LS" {
		//        Test error exits for the least squares driver routines.
		//
		//        DGELS
		*srnamt = "DGELS "
		*infot = 1
		golapack.Dgels('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgels('N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgels('N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgels('N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgels('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgels('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgels('N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELS ", &info, lerr, ok, t)

		//        DGELSS
		*srnamt = "DGELSS"
		*infot = 1
		golapack.Dgelss(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELSS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgelss(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELSS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgelss(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELSS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgelss(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGELSS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgelss(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGELSS", &info, lerr, ok, t)

		//        DGELSY
		*srnamt = "DGELSY"
		*infot = 1
		golapack.Dgelsy(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgelsy(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgelsy(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgelsy(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgelsy(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgelsy(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGELSY", &info, lerr, ok, t)

		//        DGELSD
		*srnamt = "DGELSD"
		*infot = 1
		golapack.Dgelsd(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgelsd(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgelsd(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgelsd(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgelsd(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgelsd(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("DGELSD", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
