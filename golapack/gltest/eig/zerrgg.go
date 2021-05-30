package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrgg tests the error exits for ZGGES, ZGGESX, ZGGEV, ZGGEVX,
// ZGGES3, ZGGEV3, ZGGGLM, ZGGHRD, ZGGLSE, ZGGQRF, ZGGRQF,
// ZGGSVD3, ZGGSVP3, ZHGEQZ, ZTGEVC, ZTGEXC, ZTGSEN, ZTGSJA,
// ZTGSNA, ZTGSYL, and ZUNCSD.
func Zerrgg(path []byte, t *testing.T) {
	var anrm, bnrm, dif, one, scale, tola, tolb, zero float64
	var dummyk, dummyl, i, ifst, ihi, ilo, ilst, info, j, lw, lwork, m, ncycle, nmax, nt, sdim int

	nmax = 3
	lw = 6 * nmax
	one = 1.0
	zero = 0.0
	bw := make([]bool, 3)
	sel := make([]bool, 3)
	alpha := cvf(3)
	beta := cvf(3)
	tau := cvf(3)
	w := cvf(lw)
	ls := vf(3)
	r1 := vf(3)
	r2 := vf(3)
	rce := vf(3)
	rcv := vf(3)
	rs := vf(3)
	rw := vf(lw)
	idum := make([]int, 3)
	iw := make([]int, lw)
	a := cmf(3, 3, opts)
	b := cmf(3, 3, opts)
	q := cmf(3, 3, opts)
	u := cmf(3, 3, opts)
	v := cmf(3, 3, opts)
	z := cmf(3, 3, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		sel[j-1] = true
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, zero)
			b.SetRe(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.SetRe(i-1, i-1, one)
		b.SetRe(i-1, i-1, one)
	}
	(*ok) = true
	tola = 1.0
	tolb = 1.0
	ifst = 1
	ilst = 1
	nt = 0
	lwork = 1

	//     Test error exits for the GG path.
	if string(c2) == "GG" {
		//        ZGGHRD
		*srnamt = "ZGGHRD"
		*infot = 1
		golapack.Zgghrd('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgghrd('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgghrd('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgghrd('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgghrd('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgghrd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgghrd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zgghrd('V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zgghrd('N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGHRD", &info, lerr, ok, t)
		nt = nt + 9

		//        ZGGHD3
		*srnamt = "ZGGHD3"
		*infot = 1
		golapack.Zgghd3('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgghd3('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgghd3('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgghd3('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgghd3('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgghd3('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgghd3('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 11
		golapack.Zgghd3('V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		*infot = 13
		golapack.Zgghd3('N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, &lw, &info)
		Chkxer("ZGGHD3", &info, lerr, ok, t)
		nt = nt + 9

		//        ZHGEQZ
		*srnamt = "ZHGEQZ"
		*infot = 1
		golapack.Zhgeqz('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhgeqz('E', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhgeqz('E', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhgeqz('E', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhgeqz('E', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhgeqz('E', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhgeqz('E', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhgeqz('E', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 14
		golapack.Zhgeqz('E', 'V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		*infot = 16
		golapack.Zhgeqz('E', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHGEQZ", &info, lerr, ok, t)
		nt = nt + 10

		//        ZTGEVC
		*srnamt = "ZTGEVC"
		*infot = 1
		golapack.Ztgevc('/', 'A', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztgevc('R', '/', sel, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztgevc('R', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztgevc('L', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 12
		golapack.Ztgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		*infot = 13
		golapack.Ztgevc('R', 'A', sel, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &m, w, rw, &info)
		Chkxer("ZTGEVC", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the GSV path.
	} else if string(path) == "GSV" {
		//        ZGGSVD3
		*srnamt = "ZGGSVD3"
		*infot = 1
		golapack.Zggsvd3('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggsvd3('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggsvd3('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 4
		golapack.Zggsvd3('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggsvd3('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 6
		golapack.Zggsvd3('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 10
		golapack.Zggsvd3('N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 12
		golapack.Zggsvd3('N', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 16
		golapack.Zggsvd3('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 18
		golapack.Zggsvd3('N', 'V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, u, func() *int { y := 2; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		*infot = 20
		golapack.Zggsvd3('N', 'N', 'Q', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &dummyk, &dummyl, a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), r1, r2, u, func() *int { y := 2; return &y }(), v, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), w, &lwork, rw, &idum, &info)
		Chkxer("ZGGSVD3", &info, lerr, ok, t)
		nt = nt + 11

		//        ZGGSVP3
		*srnamt = "ZGGSVP3"
		*infot = 1
		golapack.Zggsvp3('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggsvp3('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggsvp3('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 4
		golapack.Zggsvp3('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggsvp3('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 6
		golapack.Zggsvp3('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 8
		golapack.Zggsvp3('N', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 10
		golapack.Zggsvp3('N', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 16
		golapack.Zggsvp3('U', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 18
		golapack.Zggsvp3('N', 'V', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 2; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		*infot = 20
		golapack.Zggsvp3('N', 'N', 'Q', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &tola, &tolb, &dummyk, &dummyl, u, func() *int { y := 2; return &y }(), v, func() *int { y := 2; return &y }(), q, func() *int { y := 1; return &y }(), &iw, rw, tau, w, &lwork, &info)
		Chkxer("ZGGSVP3", &info, lerr, ok, t)
		nt = nt + 11

		//        ZTGSJA
		*srnamt = "ZTGSJA"
		*infot = 1
		golapack.Ztgsja('/', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztgsja('N', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztgsja('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztgsja('N', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 12
		golapack.Ztgsja('N', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 18
		golapack.Ztgsja('U', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 0; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 20
		golapack.Ztgsja('N', 'V', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		*infot = 22
		golapack.Ztgsja('N', 'N', 'Q', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &dummyk, &dummyl, a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &tola, &tolb, r1, r2, u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), w, &ncycle, &info)
		Chkxer("ZTGSJA", &info, lerr, ok, t)
		nt = nt + 11

		//     Test error exits for the GLM path.
	} else if string(path) == "GLM" {
		//        ZGGGLM
		*srnamt = "ZGGGLM"
		*infot = 1
		golapack.Zggglm(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggglm(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggglm(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggglm(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 7
		golapack.Zggglm(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		*infot = 12
		golapack.Zggglm(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGGLM", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the LSE path.
	} else if string(path) == "LSE" {
		//        ZGGLSE
		*srnamt = "ZGGLSE"
		*infot = 1
		golapack.Zgglse(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgglse(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgglse(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgglse(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), tau, alpha, beta, w, &lw, &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		*infot = 12
		golapack.Zgglse(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), tau, alpha, beta, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGLSE", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the CSD path.
	} else if string(path) == "CSD" {
		//        ZUNCSD
		*srnamt = "ZUNCSD"
		*infot = 7
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 8
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 20
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 22
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 24
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, toPtr(-1), a, func() *int { y := 1; return &y }(), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		*infot = 26
		golapack.Zuncsd('Y', 'Y', 'Y', 'Y', 'N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), rs, a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a, toPtr(-1), w, &lw, rw, &lw, &iw, &info)
		Chkxer("ZUNCSD", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the GQR path.
	} else if string(path) == "GQR" {
		//        ZGGQRF
		*srnamt = "ZGGQRF"
		*infot = 1
		golapack.Zggqrf(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggqrf(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		*infot = 8
		golapack.Zggqrf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 0; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggqrf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGQRF", &info, lerr, ok, t)
		nt = nt + 6

		//        ZGGRQF
		*srnamt = "ZGGRQF"
		*infot = 1
		golapack.Zggrqf(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggrqf(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		*infot = 8
		golapack.Zggrqf(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 0; return &y }(), beta, w, &lw, &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggrqf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), alpha, b, func() *int { y := 1; return &y }(), beta, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGGRQF", &info, lerr, ok, t)
		nt = nt + 6

		//     Test error exits for the ZGS, ZGV, ZGX, and ZXV paths.
	} else if string(path) == "ZGS" || string(path) == "ZGV" || string(path) == "ZGX" || string(path) == "ZXV" {
		//        ZGGES
		*srnamt = "ZGGES "
		*infot = 1
		golapack.Zgges('/', 'N', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgges('N', '/', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgges('N', 'V', '/', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgges('N', 'V', 'S', Zlctes, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgges('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgges('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgges('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgges('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgges('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgges('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		*infot = 18
		golapack.Zgges('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES ", &info, lerr, ok, t)
		nt = nt + 11

		//        ZGGES3
		*srnamt = "ZGGES3"
		*infot = 1
		golapack.Zgges3('/', 'N', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgges3('N', '/', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgges3('N', 'V', '/', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgges3('N', 'V', 'S', Zlctes, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgges3('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgges3('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgges3('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgges3('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgges3('N', 'V', 'S', Zlctes, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgges3('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		*infot = 18
		golapack.Zgges3('V', 'V', 'S', Zlctes, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &bw, &info)
		Chkxer("ZGGES3", &info, lerr, ok, t)
		nt = nt + 11

		//        ZGGESX
		*srnamt = "ZGGESX"
		*infot = 1
		golapack.Zggesx('/', 'N', 'S', Zlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggesx('N', '/', 'S', Zlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggesx('V', 'V', '/', Zlctsx, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggesx('V', 'V', 'S', Zlctsx, '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 10
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 17
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 17
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 21
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), &sdim, alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, func() *int { y := 1; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		*infot = 24
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), &sdim, alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), rce, rcv, w, func() *int { y := 32; return &y }(), rw, &iw, func() *int { y := 0; return &y }(), &bw, &info)
		Chkxer("ZGGESX", &info, lerr, ok, t)
		nt = nt + 13

		//        ZGGEV
		*srnamt = "ZGGEV"
		*infot = 1
		golapack.Zggev('/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggev('N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggev('V', 'V', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 7
		golapack.Zggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggev('N', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggev('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggev('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggev('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV", &info, lerr, ok, t)
		nt = nt + 10

		//        ZGGEV3
		*srnamt = "ZGGEV3"
		*infot = 1
		golapack.Zggev3('/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggev3('N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggev3('V', 'V', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 7
		golapack.Zggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggev3('N', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 11
		golapack.Zggev3('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggev3('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggev3('V', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggev3('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGGEV3", &info, lerr, ok, t)
		nt = nt + 10

		//        ZGGEVX
		*srnamt = "ZGGEVX"
		*infot = 1
		golapack.Zggevx('/', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zggevx('N', '/', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zggevx('N', 'N', '/', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zggevx('N', 'N', 'N', '/', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Zggevx('N', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Zggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zggevx('N', 'V', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 2; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zggevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 1; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 1; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		*infot = 25
		golapack.Zggevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), alpha, beta, q, func() *int { y := 2; return &y }(), u, func() *int { y := 2; return &y }(), &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv, w, func() *int { y := 0; return &y }(), rw, &iw, &bw, &info)
		Chkxer("ZGGEVX", &info, lerr, ok, t)
		nt = nt + 12

		//        ZTGEXC
		*srnamt = "ZTGEXC"
		*infot = 3
		golapack.Ztgexc(true, true, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 9
		golapack.Ztgexc(false, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 9
		golapack.Ztgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 11
		golapack.Ztgexc(true, false, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		*infot = 11
		golapack.Ztgexc(true, true, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &ifst, &ilst, &info)
		Chkxer("ZTGEXC", &info, lerr, ok, t)
		nt = nt + 7

		//        ZTGSEN
		*srnamt = "ZTGSEN"
		*infot = 1
		golapack.Ztgsen(toPtr(-1), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 5
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 7
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 9
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 13
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 15
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 21
		golapack.Ztgsen(func() *int { y := 3; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, toPtr(-5), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 23
		golapack.Ztgsen(func() *int { y := 0; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 23
		golapack.Ztgsen(func() *int { y := 1; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		*infot = 23
		golapack.Ztgsen(func() *int { y := 5; return &y }(), true, true, sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), alpha, beta, q, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &m, &tola, &tolb, rcv, w, func() *int { y := 20; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZTGSEN", &info, lerr, ok, t)
		nt = nt + 11

		//        ZTGSNA
		*srnamt = "ZTGSNA"
		*infot = 1
		golapack.Ztgsna('/', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztgsna('B', '/', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztgsna('B', 'A', sel, toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztgsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztgsna('B', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 12
		golapack.Ztgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 15
		golapack.Ztgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 0; return &y }(), &m, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		*infot = 18
		golapack.Ztgsna('E', 'A', sel, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), r1, r2, func() *int { y := 1; return &y }(), &m, w, func() *int { y := 0; return &y }(), &iw, &info)
		Chkxer("ZTGSNA", &info, lerr, ok, t)
		nt = nt + 9

		//        ZTGSYL
		*srnamt = "ZTGSYL"
		*infot = 1
		golapack.Ztgsyl('/', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 2
		golapack.Ztgsyl('N', toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 3
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 4
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 6
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 8
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 10
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 0; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 12
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 0; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 14
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 0; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 16
		golapack.Ztgsyl('N', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 0; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 20
		golapack.Ztgsyl('N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		*infot = 20
		golapack.Ztgsyl('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), u, func() *int { y := 1; return &y }(), v, func() *int { y := 1; return &y }(), z, func() *int { y := 1; return &y }(), &scale, &dif, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZTGSYL", &info, lerr, ok, t)
		nt = nt + 12
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
