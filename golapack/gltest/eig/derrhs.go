package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrhs tests the error exits for DGEBAK, SGEBAL, SGEHRD, DORGHR,
// DORMHR, DHSEQR, SHSEIN, and DTREVC.
func Derrhs(path []byte, t *testing.T) {
	var i, ihi, ilo, info, j, lw, m, nmax, nt int
	sel := make([]bool, 3)
	ifaill := make([]int, 3)
	ifailr := make([]int, 3)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 3
	lw = (nmax+2)*(nmax+2) + nmax
	s := vf(nmax)
	tau := vf(nmax)
	w := vf(lw)
	wi := vf(nmax)
	wr := vf(nmax)
	a := mf(nmax, nmax, opts)
	c := mf(nmax, nmax, opts)
	vl := mf(nmax, nmax, opts)
	vr := mf(nmax, nmax, opts)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
		}
		wi.Set(j-1, float64(j))
		sel[j-1] = true
	}
	(*ok) = true
	nt = 0

	//     Test error exits of the nonsymmetric eigenvalue routines.
	if string(c2) == "HS" {
		//        DGEBAL
		*srnamt = "DGEBAL"
		*infot = 1
		golapack.Dgebal('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("DGEBAL", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgebal('N', toPtr(-1), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("DGEBAL", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgebal('N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &info)
		Chkxer("DGEBAL", &info, lerr, ok, t)
		nt = nt + 3

		//        DGEBAK
		*srnamt = "DGEBAK"
		*infot = 1
		golapack.Dgebak('/', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgebak('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgebak('N', 'R', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgebak('N', 'R', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgebak('N', 'R', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), s, toPtr(-1), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgebak('N', 'R', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), s, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEBAK", &info, lerr, ok, t)
		nt = nt + 9

		//        DGEHRD
		*srnamt = "DGEHRD"
		*infot = 1
		golapack.Dgehrd(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgehrd(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgehrd(func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgehrd(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgehrd(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgehrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgehrd(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEHRD", &info, lerr, ok, t)
		nt = nt + 7

		//        DORGHR
		*srnamt = "DORGHR"
		*infot = 1
		golapack.Dorghr(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dorghr(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dorghr(func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorghr(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dorghr(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dorghr(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dorghr(func() *int { y := 3; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORGHR", &info, lerr, ok, t)
		nt = nt + 7

		//        DORMHR
		*srnamt = "DORMHR"
		*infot = 1
		golapack.Dormhr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dormhr('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dormhr('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dormhr('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormhr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dormhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dormhr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dormhr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormhr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dormhr('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 11
		golapack.Dormhr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 13
		golapack.Dormhr('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		*infot = 13
		golapack.Dormhr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DORMHR", &info, lerr, ok, t)
		nt = nt + 16

		//        DHSEQR
		*srnamt = "DHSEQR"
		*infot = 1
		golapack.Dhseqr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dhseqr('E', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dhseqr('E', 'N', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dhseqr('E', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dhseqr('E', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dhseqr('E', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dhseqr('E', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 7
		golapack.Dhseqr('E', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		*infot = 11
		golapack.Dhseqr('E', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DHSEQR", &info, lerr, ok, t)
		nt = nt + 9

		//        DHSEIN
		*srnamt = "DHSEIN"
		*infot = 1
		golapack.Dhsein('/', 'N', 'N', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 2
		golapack.Dhsein('R', '/', 'N', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 3
		golapack.Dhsein('R', 'N', '/', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 5
		golapack.Dhsein('R', 'N', 'N', &sel, toPtr(-1), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 7
		golapack.Dhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 2; return &y }(), func() *int { y := 4; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 11
		golapack.Dhsein('L', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 13
		golapack.Dhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		*infot = 14
		golapack.Dhsein('R', 'N', 'N', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &m, w, &ifaill, &ifailr, &info)
		Chkxer("DHSEIN", &info, lerr, ok, t)
		nt = nt + 8

		//        DTREVC
		*srnamt = "DTREVC"
		*infot = 1
		golapack.Dtrevc('/', 'A', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtrevc('L', '/', &sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtrevc('L', 'A', &sel, toPtr(-1), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtrevc('L', 'A', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), vl, func() *int { y := 2; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtrevc('L', 'A', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtrevc('R', 'A', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		*infot = 11
		golapack.Dtrevc('L', 'A', &sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), vl, func() *int { y := 2; return &y }(), vr, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &m, w, &info)
		Chkxer("DTREVC", &info, lerr, ok, t)
		nt = nt + 7
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
