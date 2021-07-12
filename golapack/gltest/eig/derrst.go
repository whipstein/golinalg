package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrst tests the error exits for DSYTRD, DORGTR, DORMTR, DSPTRD,
// DOPGTR, DOPMTR, DSTEQR, SSTERF, SSTEBZ, SSTEIN, DPTEQR, DSBTRD,
// DSYEV, SSYEVX, SSYEVD, DSBEV, SSBEVX, SSBEVD,
// DSPEV, SSPEVX, SSPEVD, DSTEV, SSTEVX, SSTEVD, and SSTEDC.
// DSYEVD_2STAGE, DSYEVR_2STAGE, DSYEVX_2STAGE,
// DSYEV_2STAGE, DSBEV_2STAGE, DSBEVD_2STAGE,
// DSBEVX_2STAGE, DSYTRD_2STAGE, DSYTRD_SY2SB,
// DSYTRD_SB2ST
func Derrst(path []byte, t *testing.T) {
	var i, info, j, liw, lw, m, n, nmax, nsplit, nt int

	//     NMAX has to be at least 3 or LIW may be too small
	nmax = 3
	liw = 12 * nmax
	lw = 20 * nmax
	d := vf(3)
	e := vf(3)
	r := vf(3)
	tau := vf(3)
	w := vf(lw)
	x := vf(3)
	i1 := make([]int, 3)
	i2 := make([]int, 3)
	i3 := make([]int, 3)
	iw := make([]int, liw)
	a := mf(3, 3, opts)
	c := mf(3, 3, opts)
	q := mf(3, 3, opts)
	z := mf(3, 3, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
		}
	}
	for j = 1; j <= nmax; j++ {
		d.Set(j-1, float64(j))
		e.Set(j-1, 0.0)
		i1[j-1] = j
		i2[j-1] = j
		tau.Set(j-1, 1.)
	}
	*ok = true
	nt = 0

	//     Test error exits for the ST path.
	if string(c2) == "ST" {
		//        DSYTRD
		*srnamt = "DSYTRD"
		*infot = 1
		golapack.Dsytrd('/', toPtr(0), a, toPtr(1), d, e, tau, w, toPtr(1), &info)
		Chkxer("DSYTRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytrd('U', toPtr(-1), a, toPtr(1), d, e, tau, w, toPtr(1), &info)
		Chkxer("DSYTRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsytrd('U', toPtr(2), a, toPtr(1), d, e, tau, w, toPtr(1), &info)
		Chkxer("DSYTRD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsytrd('U', toPtr(0), a, toPtr(1), d, e, tau, w, toPtr(0), &info)
		Chkxer("DSYTRD", &info, lerr, ok, t)
		nt = nt + 4

		//        DSYTRD_2STAGE
		*srnamt = "DSYTRD_2STAGE"
		*infot = 1
		golapack.Dsytrd2stage('/', 'U', toPtr(0), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsytrd2stage('H', 'U', toPtr(0), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsytrd2stage('N', '/', toPtr(0), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsytrd2stage('N', 'U', toPtr(-1), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsytrd2stage('N', 'U', toPtr(2), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsytrd2stage('N', 'U', toPtr(0), a, toPtr(1), d, e, tau, d, toPtr(0), w, toPtr(1), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsytrd2stage('N', 'U', toPtr(0), a, toPtr(1), d, e, tau, d, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYTRD_2STAGE", &info, lerr, ok, t)
		nt = nt + 7

		//        DSYTRD_SY2SB
		*srnamt = "DSYTRD_SY2SB"
		*infot = 1
		golapack.DsytrdSy2sb('/', toPtr(0), toPtr(0), a, toPtr(1), c, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrdSy2sb('U', toPtr(-1), toPtr(0), a, toPtr(1), c, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrdSy2sb('U', toPtr(0), toPtr(-1), a, toPtr(1), c, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrdSy2sb('U', toPtr(2), toPtr(0), a, toPtr(1), c, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrdSy2sb('U', toPtr(0), toPtr(2), a, toPtr(1), c, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		*infot = 10
		golapack.DsytrdSy2sb('U', toPtr(0), toPtr(0), a, toPtr(1), c, toPtr(1), tau, w, toPtr(0), &info)
		Chkxer("DSYTRD_SY2SB", &info, lerr, ok, t)
		nt = nt + 6

		//        DSYTRD_SB2ST
		*srnamt = "DSYTRD_SB2ST"
		*infot = 1
		golapack.DsytrdSb2st('/', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrdSb2st('Y', '/', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrdSb2st('Y', 'H', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrdSb2st('Y', 'N', '/', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrdSb2st('Y', 'N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrdSb2st('Y', 'N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrdSb2st('Y', 'N', 'U', toPtr(0), toPtr(1), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 11
		golapack.DsytrdSb2st('Y', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(0), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 13
		golapack.DsytrdSb2st('Y', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		nt = nt + 9

		//        DORGTR
		*srnamt = "DORGTR"
		*infot = 1
		golapack.Dorgtr('/', toPtr(0), a, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DORGTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dorgtr('U', toPtr(-1), a, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DORGTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dorgtr('U', toPtr(2), a, toPtr(1), tau, w, toPtr(1), &info)
		Chkxer("DORGTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Dorgtr('U', toPtr(3), a, toPtr(3), tau, w, toPtr(1), &info)
		Chkxer("DORGTR", &info, lerr, ok, t)
		nt = nt + 4

		//        DORMTR
		*srnamt = "DORMTR"
		*infot = 1
		golapack.Dormtr('/', 'U', 'N', toPtr(0), toPtr(0), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dormtr('L', '/', 'N', toPtr(0), toPtr(0), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dormtr('L', 'U', '/', toPtr(0), toPtr(0), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dormtr('L', 'U', 'N', toPtr(-1), toPtr(0), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dormtr('L', 'U', 'N', toPtr(0), toPtr(-1), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Dormtr('L', 'U', 'N', toPtr(2), toPtr(0), a, toPtr(1), tau, c, toPtr(2), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 7
		golapack.Dormtr('R', 'U', 'N', toPtr(0), toPtr(2), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 10
		golapack.Dormtr('L', 'U', 'N', toPtr(2), toPtr(0), a, toPtr(2), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 12
		golapack.Dormtr('L', 'U', 'N', toPtr(0), toPtr(2), a, toPtr(1), tau, c, toPtr(1), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		*infot = 12
		golapack.Dormtr('R', 'U', 'N', toPtr(2), toPtr(0), a, toPtr(1), tau, c, toPtr(2), w, toPtr(1), &info)
		Chkxer("DORMTR", &info, lerr, ok, t)
		nt = nt + 10

		//        DSPTRD
		*srnamt = "DSPTRD"
		*infot = 1
		golapack.Dsptrd('/', toPtr(0), r, d, e, tau, &info)
		Chkxer("DSPTRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsptrd('U', toPtr(-1), r, d, e, tau, &info)
		Chkxer("DSPTRD", &info, lerr, ok, t)
		nt = nt + 2

		//        DOPGTR
		*srnamt = "DOPGTR"
		*infot = 1
		golapack.Dopgtr('/', toPtr(0), d, tau, z, toPtr(1), w, &info)
		Chkxer("DOPGTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dopgtr('U', toPtr(-1), d, tau, z, toPtr(1), w, &info)
		Chkxer("DOPGTR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dopgtr('U', toPtr(2), d, tau, z, toPtr(1), w, &info)
		Chkxer("DOPGTR", &info, lerr, ok, t)
		nt = nt + 3

		//        DOPMTR
		*srnamt = "DOPMTR"
		*infot = 1
		golapack.Dopmtr('/', 'U', 'N', toPtr(0), toPtr(0), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dopmtr('L', '/', 'N', toPtr(0), toPtr(0), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dopmtr('L', 'U', '/', toPtr(0), toPtr(0), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dopmtr('L', 'U', 'N', toPtr(-1), toPtr(0), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		*infot = 5
		golapack.Dopmtr('L', 'U', 'N', toPtr(0), toPtr(-1), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		*infot = 9
		golapack.Dopmtr('L', 'U', 'N', toPtr(2), toPtr(0), d, tau, c, toPtr(1), w, &info)
		Chkxer("DOPMTR", &info, lerr, ok, t)
		nt = nt + 6

		//        DPTEQR
		*srnamt = "DPTEQR"
		*infot = 1
		golapack.Dpteqr('/', toPtr(0), d, e, z, toPtr(1), w, &info)
		Chkxer("DPTEQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpteqr('N', toPtr(-1), d, e, z, toPtr(1), w, &info)
		Chkxer("DPTEQR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpteqr('V', toPtr(2), d, e, z, toPtr(1), w, &info)
		Chkxer("DPTEQR", &info, lerr, ok, t)
		nt = nt + 3

		//        DSTEBZ
		*srnamt = "DSTEBZ"
		*infot = 1
		golapack.Dstebz('/', 'E', toPtr(0), toPtrf64(0.0), toPtrf64(1.0), toPtr(1), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstebz('A', '/', toPtr(0), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dstebz('A', 'E', toPtr(-1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dstebz('V', 'E', toPtr(0), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dstebz('I', 'E', toPtr(0), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dstebz('I', 'E', toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dstebz('I', 'E', toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(0), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dstebz('I', 'E', toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), d, e, &m, &nsplit, x, &i1, &i2, w, &iw, &info)
		Chkxer("DSTEBZ", &info, lerr, ok, t)
		nt = nt + 8

		//        DSTEIN
		*srnamt = "DSTEIN"
		*infot = 1
		golapack.Dstein(toPtr(-1), d, e, toPtr(0), x, &i1, &i2, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEIN", &info, lerr, ok, t)
		*infot = 4
		golapack.Dstein(toPtr(0), d, e, toPtr(-1), x, &i1, &i2, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEIN", &info, lerr, ok, t)
		*infot = 4
		golapack.Dstein(toPtr(0), d, e, toPtr(1), x, &i1, &i2, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEIN", &info, lerr, ok, t)
		*infot = 9
		golapack.Dstein(toPtr(2), d, e, toPtr(0), x, &i1, &i2, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEIN", &info, lerr, ok, t)
		nt = nt + 4

		//        DSTEQR
		*srnamt = "DSTEQR"
		*infot = 1
		golapack.Dsteqr('/', toPtr(0), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEQR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsteqr('N', toPtr(-1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEQR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsteqr('V', toPtr(2), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEQR", &info, lerr, ok, t)
		nt = nt + 3

		//        DSTERF
		*srnamt = "DSTERF"
		*infot = 1
		golapack.Dsterf(toPtr(-1), d, e, &info)
		Chkxer("DSTERF", &info, lerr, ok, t)
		nt = nt + 1

		//        DSTEDC
		*srnamt = "DSTEDC"
		*infot = 1
		golapack.Dstedc('/', toPtr(0), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstedc('N', toPtr(-1), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 6
		golapack.Dstedc('V', toPtr(2), d, e, z, toPtr(1), w, toPtr(23), &iw, toPtr(28), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstedc('N', toPtr(1), d, e, z, toPtr(1), w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstedc('I', toPtr(2), d, e, z, toPtr(2), w, toPtr(0), &iw, toPtr(12), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstedc('V', toPtr(2), d, e, z, toPtr(2), w, toPtr(0), &iw, toPtr(28), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Dstedc('N', toPtr(1), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Dstedc('I', toPtr(2), d, e, z, toPtr(2), w, toPtr(19), &iw, toPtr(0), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		*infot = 10
		golapack.Dstedc('V', toPtr(2), d, e, z, toPtr(2), w, toPtr(23), &iw, toPtr(0), &info)
		Chkxer("DSTEDC", &info, lerr, ok, t)
		nt = nt + 9

		//        DSTEVD
		*srnamt = "DSTEVD"
		*infot = 1
		golapack.Dstevd('/', toPtr(0), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstevd('N', toPtr(-1), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 6
		golapack.Dstevd('V', toPtr(2), d, e, z, toPtr(1), w, toPtr(19), &iw, toPtr(12), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstevd('N', toPtr(1), d, e, z, toPtr(1), w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstevd('V', toPtr(2), d, e, z, toPtr(2), w, toPtr(12), &iw, toPtr(12), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dstevd('N', toPtr(0), d, e, z, toPtr(1), w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dstevd('V', toPtr(2), d, e, z, toPtr(2), w, toPtr(19), &iw, toPtr(11), &info)
		Chkxer("DSTEVD", &info, lerr, ok, t)
		nt = nt + 7

		//        DSTEV
		*srnamt = "DSTEV "
		*infot = 1
		golapack.Dstev('/', toPtr(0), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstev('N', toPtr(-1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dstev('V', toPtr(2), d, e, z, toPtr(1), w, &info)
		Chkxer("DSTEV ", &info, lerr, ok, t)
		nt = nt + 3

		//        DSTEVX
		*srnamt = "DSTEVX"
		*infot = 1
		golapack.Dstevx('/', 'A', toPtr(0), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstevx('N', '/', toPtr(0), d, e, toPtrf64(0.0), toPtrf64(1.0), toPtr(1), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dstevx('N', 'A', toPtr(-1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dstevx('N', 'V', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstevx('N', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstevx('N', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dstevx('N', 'I', toPtr(2), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dstevx('N', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dstevx('V', 'A', toPtr(2), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSTEVX", &info, lerr, ok, t)
		nt = nt + 9

		//        DSTEVR
		n = 1
		*srnamt = "DSTEVR"
		_iw2n1 := iw[2*n:]
		*infot = 1
		golapack.Dstevr('/', 'A', toPtr(0), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dstevr('V', '/', toPtr(0), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dstevr('V', 'A', toPtr(-1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 7
		golapack.Dstevr('V', 'V', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dstevr('V', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(1), toPtrf64(0.0), &m, w, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 9
		n = 2
		golapack.Dstevr('V', 'I', toPtr(2), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, w, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 14
		n = 1
		golapack.Dstevr('V', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, w, z, toPtr(0), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 17
		golapack.Dstevr('V', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, w, z, toPtr(1), &iw, x, toPtr(20*n-1), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		*infot = 19
		golapack.Dstevr('V', 'I', toPtr(1), d, e, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, w, z, toPtr(1), &iw, x, toPtr(20*n), &_iw2n1, toPtr(10*n-1), &info)
		Chkxer("DSTEVR", &info, lerr, ok, t)
		nt = nt + 9

		//        DSYEVD
		*srnamt = "DSYEVD"
		*infot = 1
		golapack.Dsyevd('/', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevd('N', '/', toPtr(0), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevd('N', 'U', toPtr(-1), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsyevd('N', 'U', toPtr(2), a, toPtr(1), x, w, toPtr(3), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevd('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevd('N', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(4), &iw, toPtr(1), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevd('V', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(20), &iw, toPtr(12), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevd('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevd('N', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(5), &iw, toPtr(0), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevd('V', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(27), &iw, toPtr(11), &info)
		Chkxer("DSYEVD", &info, lerr, ok, t)
		nt = nt + 10

		//        DSYEVD_2STAGE
		*srnamt = "DSYEVD_2STAGE"
		*infot = 1
		golapack.Dsyevd2stage('/', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsyevd2stage('V', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevd2stage('N', '/', toPtr(0), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevd2stage('N', 'U', toPtr(-1), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsyevd2stage('N', 'U', toPtr(2), a, toPtr(1), x, w, toPtr(3), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevd2stage('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevd2stage('N', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(4), &iw, toPtr(1), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 8
		//         CALL DSYEVD_2STAGE( 'V', 'U', 2, A, 2, X, W, 20, IW, 12, INFO )
		//         CALL CHKXER( 'DSYEVD_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 10
		golapack.Dsyevd2stage('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevd2stage('N', 'U', toPtr(2), a, toPtr(2), x, w, toPtr(2500), &iw, toPtr(0), &info)
		Chkxer("DSYEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 10
		//         CALL DSYEVD_2STAGE( 'V', 'U', 2, A, 2, X, W, 27, IW, 11, INFO )
		//         CALL CHKXER( 'DSYEVD_2STAGE', INFOT, NOUT, LERR, OK )
		nt = nt + 9

		//        DSYEVR
		*srnamt = "DSYEVR"
		n = 1
		*infot = 1
		golapack.Dsyevr('/', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevr('V', '/', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevr('V', 'A', '/', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsyevr('V', 'A', 'U', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsyevr('V', 'A', 'U', toPtr(2), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevr('V', 'V', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevr('V', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 10

		golapack.Dsyevr('V', 'I', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 15
		golapack.Dsyevr('V', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(0), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 18
		golapack.Dsyevr('V', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n-1), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		*infot = 20
		golapack.Dsyevr('V', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n-1), &info)
		Chkxer("DSYEVR", &info, lerr, ok, t)
		nt = nt + 11

		//        DSYEVR_2STAGE
		*srnamt = "DSYEVR_2STAGE"
		n = 1
		*infot = 1
		golapack.Dsyevr2stage('/', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsyevr2stage('V', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevr2stage('N', '/', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevr2stage('N', 'A', '/', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsyevr2stage('N', 'A', 'U', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsyevr2stage('N', 'A', 'U', toPtr(2), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevr2stage('N', 'V', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevr2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevr2stage('N', 'I', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 15
		golapack.Dsyevr2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(0), &iw, d, toPtr(26*n), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 18
		golapack.Dsyevr2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(0), &_iw2n1, toPtr(10*n), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		*infot = 20
		golapack.Dsyevr2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(1), toPtrf64(0.0), &m, r, z, toPtr(1), &iw, d, toPtr(2600*n), &_iw2n1, toPtr(0), &info)
		Chkxer("DSYEVR_2STAGE", &info, lerr, ok, t)
		nt = nt + 12

		//        DSYEV
		*srnamt = "DSYEV "
		*infot = 1
		golapack.Dsyev('/', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyev('N', '/', toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyev('N', 'U', toPtr(-1), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsyev('N', 'U', toPtr(2), a, toPtr(1), x, w, toPtr(3), &info)
		Chkxer("DSYEV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyev('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV ", &info, lerr, ok, t)
		nt = nt + 5

		//        DSYEV_2STAGE
		*srnamt = "DSYEV_2STAGE "
		*infot = 1
		golapack.Dsyev2stage('/', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsyev2stage('V', 'U', toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyev2stage('N', '/', toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyev2stage('N', 'U', toPtr(-1), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsyev2stage('N', 'U', toPtr(2), a, toPtr(1), x, w, toPtr(3), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyev2stage('N', 'U', toPtr(1), a, toPtr(1), x, w, toPtr(1), &info)
		Chkxer("DSYEV_2STAGE ", &info, lerr, ok, t)
		nt = nt + 6

		//        DSYEVX
		*srnamt = "DSYEVX"
		*infot = 1
		golapack.Dsyevx('/', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevx('N', '/', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(1.0), toPtr(1), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevx('N', 'A', '/', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		*infot = 4
		golapack.Dsyevx('N', 'A', 'U', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsyevx('N', 'A', 'U', toPtr(2), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevx('N', 'V', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevx('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevx('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevx('N', 'I', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevx('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Dsyevx('V', 'A', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		*infot = 17
		golapack.Dsyevx('V', 'A', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSYEVX", &info, lerr, ok, t)
		nt = nt + 12

		//        DSYEVX_2STAGE
		*srnamt = "DSYEVX_2STAGE"
		*infot = 1
		golapack.Dsyevx2stage('/', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsyevx2stage('V', 'A', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsyevx2stage('N', '/', 'U', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(1.0), toPtr(1), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsyevx2stage('N', 'A', '/', toPtr(0), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		*infot = 4
		golapack.Dsyevx2stage('N', 'A', 'U', toPtr(-1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(1), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsyevx2stage('N', 'A', 'U', toPtr(2), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsyevx2stage('N', 'V', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevx2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsyevx2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevx2stage('N', 'I', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsyevx2stage('N', 'I', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(8), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 15
		golapack.Dsyevx2stage('N', 'A', 'U', toPtr(2), a, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(0), w, toPtr(16), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		*infot = 17
		golapack.Dsyevx2stage('N', 'A', 'U', toPtr(1), a, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSYEVX_2STAGE", &info, lerr, ok, t)
		nt = nt + 13

		//        DSPEVD
		*srnamt = "DSPEVD"
		*infot = 1
		golapack.Dspevd('/', 'U', toPtr(0), d, x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspevd('N', '/', toPtr(0), d, x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dspevd('N', 'U', toPtr(-1), d, x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 7
		golapack.Dspevd('V', 'U', toPtr(2), d, x, z, toPtr(1), w, toPtr(23), &iw, toPtr(12), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspevd('N', 'U', toPtr(1), d, x, z, toPtr(1), w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspevd('N', 'U', toPtr(2), d, x, z, toPtr(1), w, toPtr(3), &iw, toPtr(1), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspevd('V', 'U', toPtr(2), d, x, z, toPtr(2), w, toPtr(16), &iw, toPtr(12), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dspevd('N', 'U', toPtr(1), d, x, z, toPtr(1), w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dspevd('N', 'U', toPtr(2), d, x, z, toPtr(1), w, toPtr(4), &iw, toPtr(0), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dspevd('V', 'U', toPtr(2), d, x, z, toPtr(2), w, toPtr(23), &iw, toPtr(11), &info)
		Chkxer("DSPEVD", &info, lerr, ok, t)
		nt = nt + 10

		//        DSPEV
		*srnamt = "DSPEV "
		*infot = 1
		golapack.Dspev('/', 'U', toPtr(0), d, w, z, toPtr(1), x, &info)
		Chkxer("DSPEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspev('N', '/', toPtr(0), d, w, z, toPtr(1), x, &info)
		Chkxer("DSPEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dspev('N', 'U', toPtr(-1), d, w, z, toPtr(1), x, &info)
		Chkxer("DSPEV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dspev('V', 'U', toPtr(2), d, w, z, toPtr(1), x, &info)
		Chkxer("DSPEV ", &info, lerr, ok, t)
		nt = nt + 4

		//        DSPEVX
		*srnamt = "DSPEVX"
		*infot = 1
		golapack.Dspevx('/', 'A', 'U', toPtr(0), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspevx('N', '/', 'U', toPtr(0), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dspevx('N', 'A', '/', toPtr(0), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		*infot = 4
		golapack.Dspevx('N', 'A', 'U', toPtr(-1), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dspevx('N', 'V', 'U', toPtr(1), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dspevx('N', 'I', 'U', toPtr(1), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dspevx('N', 'I', 'U', toPtr(1), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspevx('N', 'I', 'U', toPtr(2), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspevx('N', 'I', 'U', toPtr(1), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dspevx('V', 'A', 'U', toPtr(2), d, toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSPEVX", &info, lerr, ok, t)
		nt = nt + 10

		//     Test error exits for the SB path.
	} else if string(c2) == "SB" {
		//        DSBTRD
		*srnamt = "DSBTRD"
		*infot = 1
		golapack.Dsbtrd('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbtrd('N', '/', toPtr(0), toPtr(0), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbtrd('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbtrd('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsbtrd('N', 'U', toPtr(1), toPtr(1), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsbtrd('V', 'U', toPtr(2), toPtr(0), a, toPtr(1), d, e, z, toPtr(1), w, &info)
		Chkxer("DSBTRD", &info, lerr, ok, t)
		nt = nt + 6

		//        DSYTRD_SB2ST
		*srnamt = "DSYTRD_SB2ST"
		*infot = 1
		golapack.DsytrdSb2st('/', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrdSb2st('N', '/', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 2
		golapack.DsytrdSb2st('N', 'H', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 3
		golapack.DsytrdSb2st('N', 'N', '/', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 4
		golapack.DsytrdSb2st('N', 'N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 5
		golapack.DsytrdSb2st('N', 'N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 7
		golapack.DsytrdSb2st('N', 'N', 'U', toPtr(0), toPtr(1), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 11
		golapack.DsytrdSb2st('N', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(0), w, toPtr(1), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		*infot = 13
		golapack.DsytrdSb2st('N', 'N', 'U', toPtr(0), toPtr(0), a, toPtr(1), d, e, r, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYTRD_SB2ST", &info, lerr, ok, t)
		nt = nt + 9

		//        DSBEVD
		*srnamt = "DSBEVD"
		*infot = 1
		golapack.Dsbevd('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbevd('N', '/', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbevd('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbevd('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsbevd('N', 'U', toPtr(2), toPtr(1), a, toPtr(1), x, z, toPtr(1), w, toPtr(4), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsbevd('V', 'U', toPtr(2), toPtr(1), a, toPtr(2), x, z, toPtr(1), w, toPtr(25), &iw, toPtr(12), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbevd('N', 'U', toPtr(1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbevd('N', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(3), &iw, toPtr(1), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbevd('V', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(2), w, toPtr(18), &iw, toPtr(12), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevd('N', 'U', toPtr(1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevd('V', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(2), w, toPtr(25), &iw, toPtr(11), &info)
		Chkxer("DSBEVD", &info, lerr, ok, t)
		nt = nt + 11

		//        DSBEVD_2STAGE
		*srnamt = "DSBEVD_2STAGE"
		*infot = 1
		golapack.Dsbevd2stage('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsbevd2stage('V', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbevd2stage('N', '/', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbevd2stage('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbevd2stage('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsbevd2stage('N', 'U', toPtr(2), toPtr(1), a, toPtr(1), x, z, toPtr(1), w, toPtr(4), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 9
		//         CALL DSBEVD_2STAGE( 'V', 'U', 2, 1, A, 2, X, Z, 1, W,
		//     $                                      25, IW, 12, INFO )
		//         CALL CHKXER( 'DSBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 11
		golapack.Dsbevd2stage('N', 'U', toPtr(1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbevd2stage('N', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(3), &iw, toPtr(1), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 11
		//         CALL DSBEVD_2STAGE( 'V', 'U', 2, 0, A, 1, X, Z, 2, W,
		//     $                                      18, IW, 12, INFO )
		//         CALL CHKXER( 'DSBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 13
		golapack.Dsbevd2stage('N', 'U', toPtr(1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(1), &iw, toPtr(0), &info)
		Chkxer("DSBEVD_2STAGE", &info, lerr, ok, t)
		//         INFOT = 13
		//         CALL DSBEVD_2STAGE( 'V', 'U', 2, 0, A, 1, X, Z, 2, W,
		//     $                                      25, IW, 11, INFO )
		//         CALL CHKXER( 'DSBEVD_2STAGE', INFOT, NOUT, LERR, OK )
		//         NT = NT + 12
		nt = nt + 9

		//        DSBEV
		*srnamt = "DSBEV "
		*infot = 1
		golapack.Dsbev('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbev('N', '/', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbev('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbev('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsbev('N', 'U', toPtr(2), toPtr(1), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsbev('V', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, &info)
		Chkxer("DSBEV ", &info, lerr, ok, t)
		nt = nt + 6

		//        DSBEV_2STAGE
		*srnamt = "DSBEV_2STAGE "
		*infot = 1
		golapack.Dsbev2stage('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsbev2stage('V', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbev2stage('N', '/', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbev2stage('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbev2stage('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsbev2stage('N', 'U', toPtr(2), toPtr(1), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsbev2stage('N', 'U', toPtr(2), toPtr(0), a, toPtr(1), x, z, toPtr(0), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbev2stage('N', 'U', toPtr(0), toPtr(0), a, toPtr(1), x, z, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSBEV_2STAGE ", &info, lerr, ok, t)
		nt = nt + 8

		//        DSBEVX
		*srnamt = "DSBEVX"
		*infot = 1
		golapack.Dsbevx('/', 'A', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbevx('N', '/', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbevx('N', 'A', '/', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbevx('N', 'A', 'U', toPtr(-1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsbevx('N', 'A', 'U', toPtr(0), toPtr(-1), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsbevx('N', 'A', 'U', toPtr(2), toPtr(1), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dsbevx('V', 'A', 'U', toPtr(2), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(2), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsbevx('N', 'V', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsbevx('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsbevx('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevx('N', 'I', 'U', toPtr(2), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevx('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Dsbevx('V', 'A', 'U', toPtr(2), toPtr(0), a, toPtr(1), q, toPtr(2), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, &iw, &i3, &info)
		Chkxer("DSBEVX", &info, lerr, ok, t)
		nt = nt + 13

		//        DSBEVX_2STAGE
		*srnamt = "DSBEVX_2STAGE"
		*infot = 1
		golapack.Dsbevx2stage('/', 'A', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 1
		golapack.Dsbevx2stage('V', 'A', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsbevx2stage('N', '/', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsbevx2stage('N', 'A', '/', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsbevx2stage('N', 'A', 'U', toPtr(-1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsbevx2stage('N', 'A', 'U', toPtr(0), toPtr(-1), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.Dsbevx2stage('N', 'A', 'U', toPtr(2), toPtr(1), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		//         INFOT = 9
		//         CALL DSBEVX_2STAGE( 'V', 'A', 'U', 2, 0, A, 1, Q, 1, 0.0D0,
		//     $          0.0D0, 0, 0, 0.0D0, M, X, Z, 2, W, 0, IW, I3, INFO )
		//         CALL CHKXER( 'DSBEVX_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 11
		golapack.Dsbevx2stage('N', 'V', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsbevx2stage('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 12
		golapack.Dsbevx2stage('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevx2stage('N', 'I', 'U', toPtr(2), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(2), toPtr(1), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsbevx2stage('N', 'I', 'U', toPtr(1), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(1), toPtr(2), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		//         INFOT = 18
		//         CALL DSBEVX_2STAGE( 'V', 'A', 'U', 2, 0, A, 1, Q, 2, 0.0D0,
		//     $          0.0D0, 0, 0, 0.0D0, M, X, Z, 1, W, 0, IW, I3, INFO )
		//         CALL CHKXER( 'DSBEVX_2STAGE', INFOT, NOUT, LERR, OK )
		*infot = 20
		golapack.Dsbevx2stage('N', 'A', 'U', toPtr(0), toPtr(0), a, toPtr(1), q, toPtr(1), toPtrf64(0.0), toPtrf64(0.0), toPtr(0), toPtr(0), toPtrf64(0.0), &m, x, z, toPtr(1), w, toPtr(0), &iw, &i3, &info)
		Chkxer("DSBEVX_2STAGE", &info, lerr, ok, t)
		//         NT = NT + 15
		nt = nt + 13
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s routines passed the tests of the error exits (%3d tests done)\n", path, nt)
	} else {
		fmt.Printf(" *** %3s routines failed the tests of the error exits ***\n", path)
	}
}
