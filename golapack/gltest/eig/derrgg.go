package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrgg tests the error exits for DGGES, DGGESX, DGGEV,  DGGEVX,
// DGGGLM, DGGHRD, DGGLSE, DGGQRF, DGGRQF, DGGSVD3,
// DGGSVP3, DHGEQZ, DORCSD, DTGEVC, DTGEXC, DTGSEN, DTGSJA, DTGSNA,
// DGGES3, DGGEV3, and DTGSYL.
func Derrgg(path []byte, t *testing.T) {
	var anrm, bnrm, dif, one, scale, tola, tolb, zero float64
	var dummyk, dummyl, i, ifst, ihi, ilo, ilst, info, j, lw, lwork, m, ncycle, nmax, nt, sdim int

	nmax = 3
	lw = 6 * nmax
	one = 1.0
	zero = 0.0
	bw := make([]bool, 3)
	sel := make([]bool, 3)
	ls := vf(3)
	r1 := vf(3)
	r2 := vf(3)
	r3 := vf(3)
	rce := vf(2)
	rcv := vf(2)
	rs := vf(3)
	tau := vf(3)
	w := vf(lw)
	idum := make([]int, 3)
	iw := make([]int, 3)
	a := mf(3, 3, opts)
	b := mf(3, 3, opts)
	q := mf(3, 3, opts)
	u := mf(3, 3, opts)
	v := mf(3, 3, opts)
	z := mf(3, 3, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := string(path[1:3])

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		sel[j-1] = true
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, zero)
			b.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.Set(i-1, i-1, one)
		b.Set(i-1, i-1, one)
	}
	(*ok) = true
	tola = 1.0
	tolb = 1.0
	ifst = 1
	ilst = 1
	nt = 0
	lwork = 1

	//     Test error exits for the GG path.
	if c2 == "GG" {
		//        DGGHRD
		*srnamt = "DGGHRD"
		*infot = 1
		golapack.Dgghrd('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgghrd('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgghrd('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgghrd('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgghrd('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgghrd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgghrd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgghrd('V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgghrd('N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGHRD", &info, lerr, ok, t)
		nt = nt + 9

		//        DGGHD3
		*srnamt = "DGGHD3"
		*infot = 1
		golapack.Dgghd3('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgghd3('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgghd3('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgghd3('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgghd3('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgghd3('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgghd3('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgghd3('V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgghd3('N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DGGHD3", &info, lerr, ok, t)
		nt = nt + 9

		//        DHGEQZ
		*srnamt = "DHGEQZ"
		*infot = 1
		golapack.Dhgeqz('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dhgeqz('E', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dhgeqz('E', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dhgeqz('E', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dhgeqz('E', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dhgeqz('E', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 8
		golapack.Dhgeqz('E', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 10
		golapack.Dhgeqz('E', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 15
		golapack.Dhgeqz('E', 'V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dhgeqz('E', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("DHGEQZ", &info, lerr, ok, t)
		nt = nt + 10

		//        DTGEVC
		*srnamt = "DTGEVC"
		*infot = 1
		golapack.Dtgevc('/', 'A', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtgevc('R', '/', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtgevc('R', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtgevc('L', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 12
		golapack.Dtgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		*infot = 13
		golapack.Dtgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &m, w, &info)
		Chkxer("DTGEVC", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the GSV path.
	} else if string(path) == "GSV" {
		//        DGGSVD3
		*srnamt = "DGGSVD3"
		*infot = 1
		golapack.Dggsvd3('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggsvd3('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggsvd3('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 4
		golapack.Dggsvd3('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggsvd3('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 6
		golapack.Dggsvd3('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 10
		golapack.Dggsvd3('N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggsvd3('N', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggsvd3('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 18
		golapack.Dggsvd3('N', 'V', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		*infot = 20
		golapack.Dggsvd3('N', 'N', 'Q', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, &idum, &info)
		Chkxer("DGGSVD3", &info, lerr, ok, t)
		nt = nt + 11

		//        DGGSVP3
		*srnamt = "DGGSVP3"
		*infot = 1
		golapack.Dggsvp3('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggsvp3('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggsvp3('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 4
		golapack.Dggsvp3('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggsvp3('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 6
		golapack.Dggsvp3('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 8
		golapack.Dggsvp3('N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 10
		golapack.Dggsvp3('N', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggsvp3('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 18
		golapack.Dggsvp3('N', 'V', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		*infot = 20
		golapack.Dggsvp3('N', 'N', 'Q', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, tau, w, &lwork, &info)
		Chkxer("DGGSVP3", &info, lerr, ok, t)
		nt = nt + 11

		//        DTGSJA
		*srnamt = "DTGSJA"
		*infot = 1
		golapack.Dtgsja('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtgsja('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtgsja('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtgsja('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 12
		golapack.Dtgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 18
		golapack.Dtgsja('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 0; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 20
		golapack.Dtgsja('N', 'V', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		*infot = 22
		golapack.Dtgsja('N', 'N', 'Q', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), w, &ncycle, &info)
		Chkxer("DTGSJA", &info, lerr, ok, t)
		nt = nt + 11

		//     Test error exits for the GLM path.
	} else if string(path) == "GLM" {
		//        DGGGLM
		*srnamt = "DGGGLM"
		*infot = 1
		golapack.Dggglm(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggglm(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggglm(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggglm(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 7
		golapack.Dggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggglm(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGGLM", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the LSE path.
	} else if string(path) == "LSE" {
		//        DGGLSE
		*srnamt = "DGGLSE"
		*infot = 1
		golapack.Dgglse(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgglse(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgglse(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, w, &lw, &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgglse(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGLSE", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the CSD path.
	} else if string(path) == "CSD" {
		//        DORCSD
		*srnamt = "DORCSD"
		*infot = 7
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 20
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 22
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 24
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		*infot = 26
		golapack.Dorcsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.VectorIdx(0), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, toPtr(-1), w, &lw, &iw, &info)
		Chkxer("DORCSD", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the GQR path.
	} else if string(path) == "GQR" {
		//        DGGQRF
		*srnamt = "DGGQRF"
		*infot = 1
		golapack.Dggqrf(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggqrf(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		*infot = 8
		golapack.Dggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 0; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		*infot = 11
		golapack.Dggqrf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGQRF", &info, lerr, ok, t)
		nt = nt + 6

		//        DGGRQF
		*srnamt = "DGGRQF"
		*infot = 1
		golapack.Dggrqf(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggrqf(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		*infot = 8
		golapack.Dggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 0; return &y }(), r2, w, &lw, &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		*infot = 11
		golapack.Dggrqf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), r1, b, func() *int { y := 1; return &y }(), r2, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGRQF", &info, lerr, ok, t)
		nt = nt + 6

		//     Test error exits for the DGS, DGV, DGX, and DXV paths.
	} else if string(path) == "DGS" || string(path) == "DGV" || string(path) == "DGX" || string(path) == "DXV" {
		//        DGGES
		*srnamt = "DGGES "
		*infot = 1
		golapack.Dgges('/', 'N', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgges('N', '/', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgges('N', 'V', '/', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgges('N', 'V', 'S', Dlctes, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgges('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgges('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgges('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgges('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgges('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgges('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		*infot = 19
		golapack.Dgges('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES ", &info, lerr, ok, t)
		nt = nt + 11

		//        DGGES3
		*srnamt = "DGGES3 "
		*infot = 1
		golapack.Dgges3('/', 'N', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgges3('N', '/', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgges3('N', 'V', '/', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgges3('N', 'V', 'S', Dlctes, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgges3('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgges3('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgges3('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgges3('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgges3('N', 'V', 'S', Dlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgges3('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		*infot = 19
		golapack.Dgges3('V', 'V', 'S', Dlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGES3 ", &info, lerr, ok, t)
		nt = nt + 11

		//        DGGESX
		*srnamt = "DGGESX"
		*infot = 1
		golapack.Dggesx('/', 'N', 'S', Dlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggesx('N', '/', 'S', Dlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggesx('V', 'V', '/', Dlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggesx('V', 'V', 'S', Dlctsx, '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 18
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 18
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 22
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		*infot = 24
		golapack.Dggesx('V', 'V', 'S', Dlctsx, 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 32; return &y }(), &iw, func() *int { y := 0; return &y }(), &bw, &info)
		Chkxer("DGGESX", &info, lerr, ok, t)
		nt = nt + 13

		//        DGGEV
		*srnamt = "DGGEV "
		*infot = 1
		golapack.Dggev('/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggev('N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggev('V', 'V', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggev('N', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggev('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggev('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggev('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV ", &info, lerr, ok, t)
		nt = nt + 10

		//        DGGEV3
		*srnamt = "DGGEV3 "
		*infot = 1
		golapack.Dggev3('/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggev3('N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggev3('V', 'V', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggev3('N', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 12
		golapack.Dggev3('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggev3('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggev3('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGGEV3 ", &info, lerr, ok, t)
		nt = nt + 10

		//        DGGEVX
		*srnamt = "DGGEVX"
		*infot = 1
		golapack.Dggevx('/', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dggevx('N', '/', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dggevx('N', 'N', '/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dggevx('N', 'N', 'N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dggevx('N', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dggevx('N', 'V', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dggevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		*infot = 26
		golapack.Dggevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, r3, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), &iw, &bw, &info)
		Chkxer("DGGEVX", &info, lerr, ok, t)
		nt = nt + 12

		//        DTGEXC
		*srnamt = "DTGEXC"
		*infot = 3
		golapack.Dtgexc(true, true, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 9
		golapack.Dtgexc(false, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 9
		golapack.Dtgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 11
		golapack.Dtgexc(true, false, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 11
		golapack.Dtgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &ifst, &ilst, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		*infot = 15
		golapack.Dtgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("DTGEXC", &info, lerr, ok, t)
		nt = nt + 8

		//        DTGSEN
		*srnamt = "DTGSEN"
		*infot = 1
		golapack.Dtgsen(toPtr(-1), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 5
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 7
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 9
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 14
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 16
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 22
		golapack.Dtgsen(func() *int { y := 0; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 22
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 22
		golapack.Dtgsen(func() *int { y := 2; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 24
		golapack.Dtgsen(func() *int { y := 0; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 24
		golapack.Dtgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		*infot = 24
		golapack.Dtgsen(func() *int { y := 2; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, r3, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("DTGSEN", &info, lerr, ok, t)
		nt = nt + 12

		//        DTGSNA
		*srnamt = "DTGSNA"
		*infot = 1
		golapack.Dtgsna('/', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtgsna('B', '/', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtgsna('B', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtgsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtgsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 12
		golapack.Dtgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 15
		golapack.Dtgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 0; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		*infot = 18
		golapack.Dtgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 0; return &y }(), &iw, &info)
		Chkxer("DTGSNA", &info, lerr, ok, t)
		nt = nt + 9

		//        DTGSYL
		*srnamt = "DTGSYL"
		*infot = 1
		golapack.Dtgsyl('/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 2
		golapack.Dtgsyl('N', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 3
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 4
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 6
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 8
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 10
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 12
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 14
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 16
		golapack.Dtgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 20
		golapack.Dtgsyl('N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		*infot = 20
		golapack.Dtgsyl('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DTGSYL", &info, lerr, ok, t)
		nt = nt + 12
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
