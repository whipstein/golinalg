package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrls tests the error exits for the COMPLEX*16 least squares
// driver routines (ZGELS, CGELSS, CGELSY, CGELSD).
func Zerrls(path []byte, t *testing.T) {
	var rcond float64
	var info, irnk int

	w := cvf(2)
	rw := vf(2)
	s := vf(2)
	ip := make([]int, 2)
	a := cmf(2, 2, opts)
	b := cmf(2, 2, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	a.Set(0, 0, (1.0 + 0.0*1i))
	a.Set(0, 1, (2.0 + 0.0*1i))
	a.Set(1, 1, (3.0 + 0.0*1i))
	a.Set(1, 0, (4.0 + 0.0*1i))
	(*ok) = true

	//     Test error exits for the least squares driver routines.
	if string(c2) == "LS" {
		//        ZGELS
		*srnamt = "ZGELS "
		*infot = 1
		golapack.Zgels('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgels('N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgels('N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgels('N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgels('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zgels('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zgels('N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGELS ", &info, lerr, ok, t)

		//        ZGELSS
		*srnamt = "ZGELSS"
		*infot = 1
		golapack.Zgelss(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGELSS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgelss(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGELSS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgelss(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGELSS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgelss(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 2; return &y }(), rw, &info)
		Chkxer("ZGELSS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgelss(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 2; return &y }(), rw, &info)
		Chkxer("ZGELSS", &info, lerr, ok, t)

		//        ZGELSY
		*srnamt = "ZGELSY"
		*infot = 1
		golapack.Zgelsy(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgelsy(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgelsy(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgelsy(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgelsy(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)
		*infot = 12
		golapack.Zgelsy(func() *int { y := 0; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 3; return &y }(), &ip, &rcond, &irnk, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGELSY", &info, lerr, ok, t)

		//        ZGELSD
		*srnamt = "ZGELSD"
		*infot = 1
		golapack.Zgelsd(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgelsd(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgelsd(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgelsd(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgelsd(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), s, &rcond, &irnk, w, func() *int { y := 10; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
		*infot = 12
		golapack.Zgelsd(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), s, &rcond, &irnk, w, func() *int { y := 1; return &y }(), rw, &ip, &info)
		Chkxer("ZGELSD", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
