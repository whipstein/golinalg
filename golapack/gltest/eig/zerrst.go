package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrst tests the error exits for ZHETRD, ZUNGTR, CUNMTR, ZHPTRD,
// ZUNGTR, ZUPMTR, ZSTEQR, CSTEIN, ZPTEQR, ZHBTRD,
// ZHEEV, CHEEVX, CHEEVD, ZHBEV, CHBEVX, CHBEVD,
// ZHPEV, CHPEVX, CHPEVD, and ZSTEDC.
// ZHEEVD_2STAGE, ZHEEVR_2STAGE, ZHEEVX_2STAGE,
// ZHEEV_2STAGE, ZHBEV_2STAGE, ZHBEVD_2STAGE,
// ZHBEVX_2STAGE, ZHETRD_2STAGE
func Zerrst(path []byte, t *testing.T) {
	var i, info, j, liw, lw, m, n, nmax, nt int

	nmax = 3
	liw = 12 * nmax
	lw = 20 * nmax
	tau := cvf(3)
	w := cvf(lw)
	d := vf(3)
	e := vf(3)
	r := vf(lw)
	rw := vf(lw)
	x := vf(3)
	i1 := make([]int, 3)
	i2 := make([]int, 3)
	i3 := make([]int, 3)
	iw := make([]int, liw)
	a := cmf(3, 3, opts)
	c := cmf(3, 3, opts)
	q := cmf(3, 3, opts)
	z := cmf(3, 3, opts)
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
	}
	for j = 1; j <= nmax; j++ {
		d.Set(j-1, float64(j))
		e.Set(j-1, 0.0)
		i1[j-1] = j
		i2[j-1] = j
		tau.Set(j-1, 1.)
	}
	(*ok) = true
	nt = 0

	//     Test error exits for the ST path.
	if string(c2) == "ST" {
		//        ZHETRD
		*srnamt = "ZHETRD"
		*infot = 1
		golapack.Zhetrd('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrd('U', toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhetrd('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhetrd('U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRD", &info, lerr, ok, t)
		nt = nt + 4

		//        ZHETRD_2STAGE
		*srnamt = "ZHETRD_2STAGE"
		*infot = 1
		golapack.Zhetrd2stage('/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Zhetrd2stage('H', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrd2stage('N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhetrd2stage('N', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhetrd2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhetrd2stage('N', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Zhetrd2stage('N', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, tau, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRD_2STAGE", &info, lerr, ok, t)
		nt = nt + 7

		//        ZHETRD_HE2HB
		*srnamt = "ZHETRD_HE2HB"
		*infot = 1
		golapack.Zhetrdhe2hb('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrdhe2hb('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhetrdhe2hb('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhetrdhe2hb('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhetrdhe2hb('U', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhetrdhe2hb('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRD_HE2HB", &info, lerr, ok, t)
		nt = nt + 6

		//        ZHETRD_HB2ST
		*srnamt = "ZHETRD_HB2ST"
		*infot = 1
		golapack.Zhetrdhb2st('/', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrdhb2st('Y', '/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrdhb2st('Y', 'H', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhetrdhb2st('Y', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhetrdhb2st('Y', 'N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhetrdhb2st('Y', 'N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhetrdhb2st('Y', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhetrdhb2st('Y', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhetrdhb2st('Y', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		nt = nt + 9

		//        ZUNGTR
		*srnamt = "ZUNGTR"
		*infot = 1
		golapack.Zungtr('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zungtr('U', toPtr(-1), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zungtr('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Zungtr('U', func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNGTR", &info, lerr, ok, t)
		nt = nt + 4

		//        ZUNMTR
		*srnamt = "ZUNMTR"
		*infot = 1
		golapack.Zunmtr('/', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zunmtr('L', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zunmtr('L', 'U', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zunmtr('L', 'U', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 5
		golapack.Zunmtr('L', 'U', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Zunmtr('L', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Zunmtr('R', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 10
		golapack.Zunmtr('L', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 12
		golapack.Zunmtr('L', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		*infot = 12
		golapack.Zunmtr('R', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, c, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZUNMTR", &info, lerr, ok, t)
		nt = nt + 10

		//        ZHPTRD
		*srnamt = "ZHPTRD"
		*infot = 1
		golapack.Zhptrd('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), d, e, tau, &info)
		Chkxer("ZHPTRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhptrd('U', toPtr(-1), a.CVector(0, 0), d, e, tau, &info)
		Chkxer("ZHPTRD", &info, lerr, ok, t)
		nt = nt + 2

		//        ZUPGTR
		*srnamt = "ZUPGTR"
		*infot = 1
		golapack.Zupgtr('/', func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPGTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zupgtr('U', toPtr(-1), a.CVector(0, 0), tau, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPGTR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zupgtr('U', func() *int { y := 2; return &y }(), a.CVector(0, 0), tau, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPGTR", &info, lerr, ok, t)
		nt = nt + 3

		//        ZUPMTR
		*srnamt = "ZUPMTR"
		*infot = 1
		golapack.Zupmtr('/', 'U', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zupmtr('L', '/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zupmtr('L', 'U', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zupmtr('L', 'U', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		*infot = 5
		golapack.Zupmtr('L', 'U', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		*infot = 9
		golapack.Zupmtr('L', 'U', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), tau, c, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZUPMTR", &info, lerr, ok, t)
		nt = nt + 6

		//        ZPTEQR
		*srnamt = "ZPTEQR"
		*infot = 1
		golapack.Zpteqr('/', func() *int { y := 0; return &y }(), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZPTEQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpteqr('N', toPtr(-1), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZPTEQR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpteqr('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZPTEQR", &info, lerr, ok, t)
		nt = nt + 3

		//        ZSTEIN
		*srnamt = "ZSTEIN"
		*infot = 1
		golapack.Zstein(toPtr(-1), d, e, func() *int { y := 0; return &y }(), x, &i1, &i2, z, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZSTEIN", &info, lerr, ok, t)
		*infot = 4
		golapack.Zstein(func() *int { y := 0; return &y }(), d, e, toPtr(-1), x, &i1, &i2, z, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZSTEIN", &info, lerr, ok, t)
		*infot = 4
		golapack.Zstein(func() *int { y := 0; return &y }(), d, e, func() *int { y := 1; return &y }(), x, &i1, &i2, z, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZSTEIN", &info, lerr, ok, t)
		*infot = 9
		golapack.Zstein(func() *int { y := 2; return &y }(), d, e, func() *int { y := 0; return &y }(), x, &i1, &i2, z, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZSTEIN", &info, lerr, ok, t)
		nt = nt + 4

		//        ZSTEQR
		*srnamt = "ZSTEQR"
		*infot = 1
		golapack.Zsteqr('/', func() *int { y := 0; return &y }(), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSTEQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsteqr('N', toPtr(-1), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSTEQR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zsteqr('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSTEQR", &info, lerr, ok, t)
		nt = nt + 3

		//        ZSTEDC
		*srnamt = "ZSTEDC"
		*infot = 1
		golapack.Zstedc('/', func() *int { y := 0; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 2
		golapack.Zstedc('N', toPtr(-1), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 6
		golapack.Zstedc('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 23; return &y }(), &iw, func() *int { y := 28; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 8
		golapack.Zstedc('N', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 8
		golapack.Zstedc('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), rw, func() *int { y := 23; return &y }(), &iw, func() *int { y := 28; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Zstedc('N', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Zstedc('I', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Zstedc('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 2; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 28; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 12
		golapack.Zstedc('N', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 12
		golapack.Zstedc('I', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 23; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		*infot = 12
		golapack.Zstedc('V', func() *int { y := 2; return &y }(), d, e, z, func() *int { y := 2; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 23; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSTEDC", &info, lerr, ok, t)
		nt = nt + 11

		//        ZHEEVD
		*srnamt = "ZHEEVD"
		*infot = 1
		golapack.Zheevd('/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevd('N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevd('N', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 5
		golapack.Zheevd('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevd('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevd('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevd('V', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevd('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevd('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevd('V', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 8; return &y }(), rw, func() *int { y := 18; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 12
		golapack.Zheevd('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		*infot = 12
		golapack.Zheevd('V', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 8; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 11; return &y }(), &info)
		Chkxer("ZHEEVD", &info, lerr, ok, t)
		nt = nt + 12

		//        ZHEEVD_2STAGE
		*srnamt = "ZHEEVD_2STAGE"
		*infot = 1
		golapack.Zheevd2stage('/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Zheevd2stage('V', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevd2stage('N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevd2stage('N', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zheevd2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevd2stage('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevd2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevd2stage('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 10

		golapack.Zheevd2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 700; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Zheevd2stage('N', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHEEVD_2STAGE", &info, lerr, ok, t)
		*infot = 12
		nt = nt + 10

		//        ZHEEV
		*srnamt = "ZHEEV "
		*infot = 1
		golapack.Zheev('/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheev('N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheev('N', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zheev('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, &info)
		Chkxer("ZHEEV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheev('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), rw, &info)
		Chkxer("ZHEEV ", &info, lerr, ok, t)
		nt = nt + 5

		//        ZHEEV_2STAGE
		*srnamt = "ZHEEV_2STAGE "
		*infot = 1
		golapack.Zheev2stage('/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		*infot = 1
		golapack.Zheev2stage('V', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheev2stage('N', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheev2stage('N', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zheev2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 3; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheev2stage('N', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), rw, &info)
		Chkxer("ZHEEV_2STAGE ", &info, lerr, ok, t)
		nt = nt + 6

		//        ZHEEVX
		*srnamt = "ZHEEVX"
		*infot = 1
		golapack.Zheevx('/', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevx('V', '/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevx('V', 'A', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		*infot = 4
		golapack.Zheevx('V', 'A', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zheevx('V', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevx('V', 'V', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zheevx('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevx('V', 'I', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zheevx('V', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		*infot = 17
		golapack.Zheevx('V', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, &iw, &i1, &info)
		Chkxer("ZHEEVX", &info, lerr, ok, t)
		nt = nt + 10

		//        ZHEEVX_2STAGE
		*srnamt = "ZHEEVX_2STAGE"
		*infot = 1
		golapack.Zheevx2stage('/', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Zheevx2stage('V', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevx2stage('N', '/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevx2stage('N', 'A', '/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		*infot = 4
		golapack.Zheevx2stage('N', 'A', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Zheevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevx2stage('N', 'V', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Zheevx2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevx2stage('N', 'I', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 15
		golapack.Zheevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 0; return &y }(), w, func() *int { y := 3; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		*infot = 17
		golapack.Zheevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i1, &info)
		Chkxer("ZHEEVX_2STAGE", &info, lerr, ok, t)
		nt = nt + 11

		//        ZHEEVR
		*srnamt = "ZHEEVR"
		n = 1
		*infot = 1
		golapack.Zheevr('/', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevr('V', '/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevr('V', 'A', '/', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 4
		golapack.Zheevr('V', 'A', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 6
		golapack.Zheevr('V', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevr('V', 'V', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 9
		golapack.Zheevr('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 10

		golapack.Zheevr('V', 'I', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 15
		golapack.Zheevr('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 0; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 18
		golapack.Zheevr('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n-1), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 20
		golapack.Zheevr('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n-1), toSlice(&iw, 2*n-1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		*infot = 22
		golapack.Zheevr('V', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), &iw, toPtr(10*n-1), &info)
		Chkxer("ZHEEVR", &info, lerr, ok, t)
		nt = nt + 12

		//        ZHEEVR_2STAGE
		*srnamt = "ZHEEVR_2STAGE"
		n = 1
		*infot = 1
		golapack.Zheevr2stage('/', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Zheevr2stage('V', 'A', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zheevr2stage('N', '/', 'U', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zheevr2stage('N', 'A', '/', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.Zheevr2stage('N', 'A', 'U', toPtr(-1), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Zheevr2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Zheevr2stage('N', 'V', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 15
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 0; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 18
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(2*n-1), rw, toPtr(24*n), toSlice(&iw, 2*n+1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 20
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(600*n), rw, toPtr(24*n-1), toSlice(&iw, 2*n-1-1), toPtr(10*n), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		*infot = 22
		golapack.Zheevr2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, r, z, func() *int { y := 1; return &y }(), &iw, q.CVector(0, 0), toPtr(600*n), rw, toPtr(24*n), &iw, toPtr(10*n-1), &info)
		Chkxer("ZHEEVR_2STAGE", &info, lerr, ok, t)
		nt = nt + 13

		//        ZHPEVD
		*srnamt = "ZHPEVD"
		*infot = 1
		golapack.Zhpevd('/', 'U', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhpevd('N', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhpevd('N', 'U', toPtr(-1), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhpevd('V', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhpevd('N', 'U', func() *int { y := 1; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhpevd('N', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhpevd('V', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhpevd('N', 'U', func() *int { y := 1; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhpevd('N', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhpevd('V', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 18; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhpevd('N', 'U', func() *int { y := 1; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhpevd('N', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhpevd('V', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 4; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZHPEVD", &info, lerr, ok, t)
		nt = nt + 13

		//        ZHPEV
		*srnamt = "ZHPEV "
		*infot = 1
		golapack.Zhpev('/', 'U', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHPEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhpev('N', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHPEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhpev('N', 'U', toPtr(-1), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHPEV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhpev('V', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHPEV ", &info, lerr, ok, t)
		nt = nt + 4

		//        ZHPEVX
		*srnamt = "ZHPEVX"
		*infot = 1
		golapack.Zhpevx('/', 'A', 'U', func() *int { y := 0; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhpevx('V', '/', 'U', func() *int { y := 0; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhpevx('V', 'A', '/', func() *int { y := 0; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhpevx('V', 'A', 'U', toPtr(-1), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhpevx('V', 'V', 'U', func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhpevx('V', 'I', 'U', func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhpevx('V', 'I', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Zhpevx('V', 'A', 'U', func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHPEVX", &info, lerr, ok, t)
		nt = nt + 8

		//     Test error exits for the HB path.
	} else if string(c2) == "HB" {
		//        ZHBTRD
		*srnamt = "ZHBTRD"
		*infot = 1
		golapack.Zhbtrd('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbtrd('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbtrd('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhbtrd('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhbtrd('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhbtrd('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, z, func() *int { y := 1; return &y }(), w, &info)
		Chkxer("ZHBTRD", &info, lerr, ok, t)
		nt = nt + 6

		//        ZHETRD_HB2ST
		*srnamt = "ZHETRD_HB2ST"
		*infot = 1
		golapack.Zhetrdhb2st('/', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrdhb2st('N', '/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhetrdhb2st('N', 'H', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhetrdhb2st('N', 'N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhetrdhb2st('N', 'N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhetrdhb2st('N', 'N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhetrdhb2st('N', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhetrdhb2st('N', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhetrdhb2st('N', 'N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), d, e, c.CVector(0, 0), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHETRD_HB2ST", &info, lerr, ok, t)
		nt = nt + 9

		//        ZHBEVD
		*srnamt = "ZHBEVD"
		*infot = 1
		golapack.Zhbevd('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbevd('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbevd('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhbevd('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhbevd('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhbevd('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 8; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevd('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevd('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevd('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevd('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevd('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevd('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 8; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 15
		golapack.Zhbevd('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 15
		golapack.Zhbevd('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		*infot = 15
		golapack.Zhbevd('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 8; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZHBEVD", &info, lerr, ok, t)
		nt = nt + 15

		//        ZHBEVD_2STAGE
		*srnamt = "ZHBEVD_2STAGE"
		*infot = 1
		golapack.Zhbevd2stage('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Zhbevd2stage('V', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbevd2stage('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbevd2stage('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 0; return &y }(), w, func() *int { y := 8; return &y }(), rw, func() *int { y := 25; return &y }(), &iw, func() *int { y := 12; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 11
		//         CALL golapack.ZHBEVD_2STAGE( 'V', 'U', 2, 1, A, 2, X, Z, 2,
		//     $                         W, 2, RW, 25, IW, 12, INFO )
		//         CALL CHKXER( 'ZHBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 13
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 0; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 25; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 13
		//         CALL golapack.ZHBEVD_2STAGE( 'V', 'U', 2, 1, A, 2, X, Z, 2,
		//     $                          W, 25, RW, 2, IW, 12, INFO )
		//         CALL CHKXER( 'ZHBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 15
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 15
		golapack.Zhbevd2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 25; return &y }(), rw, func() *int { y := 2; return &y }(), &iw, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 15
		//         CALL golapack.ZHBEVD_2STAGE( 'V', 'U', 2, 1, A, 2, X, Z, 2,
		//     $                          W, 25, RW, 25, IW, 2, INFO )
		//         CALL CHKXER( 'ZHBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		nt = nt + 13

		//        ZHBEV
		*srnamt = "ZHBEV "
		*infot = 1
		golapack.Zhbev('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbev('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbev('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhbev('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhbev('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhbev('V', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, rw, &info)
		Chkxer("ZHBEV ", &info, lerr, ok, t)
		nt = nt + 6

		//        ZHBEV_2STAGE
		*srnamt = "ZHBEV_2STAGE "
		*infot = 1
		golapack.Zhbev2stage('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 1
		golapack.Zhbev2stage('V', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbev2stage('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbev2stage('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhbev2stage('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhbev2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhbev2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 0; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbev2stage('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &info)
		Chkxer("ZHBEV_2STAGE ", &info, lerr, ok, t)
		nt = nt + 8

		//        ZHBEVX
		*srnamt = "ZHBEVX"
		*infot = 1
		golapack.Zhbevx('/', 'A', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbevx('V', '/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbevx('V', 'A', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		*infot = 4
		golapack.Zhbevx('V', 'A', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhbevx('V', 'A', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhbevx('V', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhbevx('V', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhbevx('V', 'V', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Zhbevx('V', 'I', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevx('V', 'I', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Zhbevx('V', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, rw, &iw, &i3, &info)
		Chkxer("ZHBEVX", &info, lerr, ok, t)
		nt = nt + 11

		//        ZHBEVX_2STAGE
		*srnamt = "ZHBEVX_2STAGE"
		*infot = 1
		golapack.Zhbevx2stage('/', 'A', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		*infot = 1
		golapack.Zhbevx2stage('V', 'A', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhbevx2stage('N', '/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhbevx2stage('N', 'A', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		*infot = 4
		golapack.Zhbevx2stage('N', 'A', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhbevx2stage('N', 'A', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhbevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		//         INFOT = 9
		//         CALL golapack.ZHBEVX_2STAGE( 'V', 'A', 'U', 2, 0, A, 1, Q, 1,
		//     $                       0.0D0, 0.0D0, 0, 0, 0.0D0,
		//     $                       M, X, Z, 2, W, 0, RW, IW, I3, INFO )
		//         CALL CHKXER( 'ZHBEVX_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 11
		golapack.Zhbevx2stage('N', 'V', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Zhbevx2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhbevx2stage('N', 'I', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 18
		golapack.Zhbevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 0; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 20
		golapack.Zhbevx2stage('N', 'A', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), q, func() *int { y := 2; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), &m, x, z, func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), rw, &iw, &i3, &info)
		Chkxer("ZHBEVX_2STAGE", &info, lerr, ok, t)
		nt = nt + 12
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
