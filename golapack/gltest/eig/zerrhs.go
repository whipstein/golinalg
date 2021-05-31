package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrhs tests the error exits for ZGEBAK, CGEBAL, CGEHRD, ZUNGHR,
// ZUNMHR, ZHSEQR, CHSEIN, and ZTREVC.
func Zerrhs(path []byte, t *testing.T) {
	var i, ihi, ilo, info, j, lw, m, nmax, nt int

	nmax = 3
	lw = nmax * nmax
	sel := make([]bool, 3)
	tau := cvf(3)
	w := cvf(lw)
	x := cvf(3)
	rw := vf(3)
	s := vf(3)
	ifaill := make([]int, 3)
	ifailr := make([]int, 3)
	a := cmf(3, 3, opts)
	c := cmf(3, 3, opts)
	vl := cmf(3, 3, opts)
	vr := cmf(3, 3, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))
		}
		sel[j-1] = true
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the nonsymmetric eigenvalue routines.
	if string(c2) == "HS" {
		//        ZGEBAL
		*srnamt = "ZGEBAL"
		(*infot) = 1
		golapack.Zgebal('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("ZGEBAL", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgebal('N', toPtr(-1), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("ZGEBAL", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgebal('N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("ZGEBAL", &info, lerr, ok, t)
		nt = nt + 3

		//        ZGEBAK
		*srnamt = "ZGEBAK"
		(*infot) = 1
		golapack.Zgebak('/', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgebak('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgebak('N', 'R', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgebak('N', 'R', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zgebak('N', 'R', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEBAK", &info, lerr, ok, t)
		nt = nt + 9

		//        ZGEHRD
		*srnamt = "ZGEHRD"
		(*infot) = 1
		golapack.Zgehrd(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgehrd(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgehrd(func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgehrd(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgehrd(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgehrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgehrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGEHRD", &info, lerr, ok, t)
		nt = nt + 7

		//        ZUNGHR
		*srnamt = "ZUNGHR"
		(*infot) = 1
		golapack.Zunghr(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zunghr(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zunghr(func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zunghr(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zunghr(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zunghr(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zunghr(func() *int { y := 3; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGHR", &info, lerr, ok, t)
		nt = nt + 7

		//        ZUNMHR
		*srnamt = "ZUNMHR"
		(*infot) = 1
		golapack.Zunmhr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zunmhr('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zunmhr('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zunmhr('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zunmhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zunmhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zunmhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zunmhr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zunmhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zunmhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zunmhr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zunmhr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zunmhr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zunmhr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 13
		golapack.Zunmhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		(*infot) = 13
		golapack.Zunmhr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMHR", &info, lerr, ok, t)
		nt = nt + 16

		//        ZHSEQR
		*srnamt = "ZHSEQR"
		(*infot) = 1
		golapack.Zhseqr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhseqr('E', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhseqr('E', 'N', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhseqr('E', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zhseqr('E', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhseqr('E', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhseqr('E', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhseqr('E', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhseqr('E', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHSEQR", &info, lerr, ok, t)
		nt = nt + 9

		//        ZHSEIN
		*srnamt = "ZHSEIN"
		(*infot) = 1
		golapack.Zhsein('/', 'N', 'N', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zhsein('R', '/', 'N', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zhsein('R', 'N', '/', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zhsein('R', 'N', 'N', &sel, toPtr(-1), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 2; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zhsein('L', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		(*infot) = 13
		golapack.Zhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &m, w, rw, &ifaill, &ifailr, &info)
		Chkxer("ZHSEIN", &info, lerr, ok, t)
		nt = nt + 8

		//        ZTREVC
		*srnamt = "ZTREVC"
		(*infot) = 1
		golapack.Ztrevc('/', 'A', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Ztrevc('L', '/', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Ztrevc('L', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Ztrevc('L', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 2; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Ztrevc('L', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Ztrevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Ztrevc('L', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 2; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &m, w, rw, &info)
		Chkxer("ZTREVC", &info, lerr, ok, t)
		nt = nt + 7
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
