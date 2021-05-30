package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrvx tests the error exits for the COMPLEX*16 driver routines
// for solving linear systems of equations.
func Zerrvx(path []byte, t *testing.T) {
	var eq byte
	var rcond float64
	var i, info, j, nmax int

	nmax = 4
	b := cvf(4)
	e := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	c := vf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	rf := vf(4)
	rw := vf(4)
	ip := make([]int, 4)
	a := cmf(4, 4, opts)
	af := cmf(4, 4, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
			af.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
		}
		b.Set(j-1, 0.)
		e.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
		c.Set(j-1, 0.)
		r.Set(j-1, 0.)
		// ip.Set(j-1, j)
	}
	eq = ' '
	(*ok) = true

	if string(c2) == "GE" {
		//        ZGESV
		*srnamt = "ZGESV "
		*infot = 1
		golapack.Zgesv(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgesv(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgesv(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGESV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgesv(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESV ", &info, lerr, ok, t)

		//        ZGESVX
		*srnamt = "ZGESVX"
		*infot = 1
		golapack.Zgesvx('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgesvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgesvx('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgesvx('N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgesvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, &eq, r, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zgesvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 10
		eq = '/'
		golapack.Zgesvx('F', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 11
		eq = 'R'
		golapack.Zgesvx('F', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 12
		eq = 'C'
		golapack.Zgesvx('F', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgesvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgesvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, &eq, r, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGESVX", &info, lerr, ok, t)

	} else if string(c2) == "GB" {
		//        ZGBSV
		*srnamt = "ZGBSV "
		*infot = 1
		golapack.Zgbsv(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbsv(func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbsv(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbsv(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbsv(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 3; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgbsv(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBSV ", &info, lerr, ok, t)

		//        ZGBSVX
		*srnamt = "ZGBSVX"
		*infot = 1
		golapack.Zgbsvx('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbsvx('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbsvx('N', 'N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgbsvx('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbsvx('N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zgbsvx('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 4; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Zgbsvx('N', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 3; return &y }(), af, func() *int { y := 3; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 12
		eq = '/'
		golapack.Zgbsvx('F', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 13
		eq = 'R'
		golapack.Zgbsvx('F', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 14
		eq = 'C'
		golapack.Zgbsvx('F', 'N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgbsvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Zgbsvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, &eq, r, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGBSVX", &info, lerr, ok, t)

	} else if string(c2) == "GT" {
		//        ZGTSV
		*srnamt = "ZGTSV "
		*infot = 1
		golapack.Zgtsv(toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgtsv(func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgtsv(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGTSV ", &info, lerr, ok, t)

		//        ZGTSVX
		*srnamt = "ZGTSVX"
		*infot = 1
		golapack.Zgtsvx('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgtsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgtsvx('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgtsvx('N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgtsvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Zgtsvx('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), a.CVector(0, 1), a.CVector(0, 2), af.CVector(0, 0), af.CVector(0, 1), af.CVector(0, 2), af.CVector(0, 3), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZGTSVX", &info, lerr, ok, t)

	} else if string(c2) == "PO" {
		//        ZPOSV
		*srnamt = "ZPOSV "
		*infot = 1
		golapack.Zposv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zposv('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zposv('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOSV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zposv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZPOSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zposv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPOSV ", &info, lerr, ok, t)

		//        ZPOSVX
		*srnamt = "ZPOSVX"
		*infot = 1
		golapack.Zposvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zposvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zposvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zposvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zposvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &eq, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zposvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 9
		eq = '/'
		golapack.Zposvx('F', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 10
		eq = 'Y'
		golapack.Zposvx('F', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Zposvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Zposvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &eq, c, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPOSVX", &info, lerr, ok, t)

	} else if string(c2) == "PP" {
		//        ZPPSV
		*srnamt = "ZPPSV "
		*infot = 1
		golapack.Zppsv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zppsv('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zppsv('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zppsv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPPSV ", &info, lerr, ok, t)

		//        ZPPSVX
		*srnamt = "ZPPSVX"
		*infot = 1
		golapack.Zppsvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zppsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zppsvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zppsvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 7
		eq = '/'
		golapack.Zppsvx('F', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 8
		eq = 'Y'
		golapack.Zppsvx('F', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Zppsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Zppsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &eq, c, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPPSVX", &info, lerr, ok, t)

	} else if string(c2) == "PB" {
		//        ZPBSV
		*srnamt = "ZPBSV "
		*infot = 1
		golapack.Zpbsv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbsv('U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbsv('U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpbsv('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zpbsv('U', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zpbsv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPBSV ", &info, lerr, ok, t)

		//        ZPBSVX
		*srnamt = "ZPBSVX"
		*infot = 1
		golapack.Zpbsvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zpbsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zpbsvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zpbsvx('N', 'U', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Zpbsvx('N', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Zpbsvx('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zpbsvx('N', 'U', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 10
		eq = '/'
		golapack.Zpbsvx('F', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 11
		eq = 'Y'
		golapack.Zpbsvx('F', 'U', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zpbsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Zpbsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &eq, c, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPBSVX", &info, lerr, ok, t)

	} else if string(c2) == "PT" {
		//        ZPTSV
		*srnamt = "ZPTSV "
		*infot = 1
		golapack.Zptsv(toPtr(-1), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zptsv(func() *int { y := 0; return &y }(), toPtr(-1), r, a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Zptsv(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZPTSV ", &info, lerr, ok, t)

		//        ZPTSVX
		*srnamt = "ZPTSVX"
		*infot = 1
		golapack.Zptsvx('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), rf, af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPTSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zptsvx('N', toPtr(-1), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), rf, af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPTSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zptsvx('N', func() *int { y := 0; return &y }(), toPtr(-1), r, a.CVector(0, 0), rf, af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPTSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zptsvx('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), rf, af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPTSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zptsvx('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), r, a.CVector(0, 0), rf, af.CVector(0, 0), b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZPTSVX", &info, lerr, ok, t)

	} else if string(c2) == "HE" {
		//        ZHESV
		*srnamt = "ZHESV "
		*infot = 1
		golapack.Zhesv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesv('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesv('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhesv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhesv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhesv('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhesv('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZHESV ", &info, lerr, ok, t)

		//        ZHESVX
		*srnamt = "ZHESVX"
		*infot = 1
		golapack.Zhesvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhesvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zhesvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhesvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhesvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zhesvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Zhesvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 3; return &y }(), rw, &info)
		Chkxer("ZHESVX", &info, lerr, ok, t)

	} else if string(c2) == "HR" {
		//        ZHESV_ROOK
		*srnamt = "ZHESV_ROOK"
		*infot = 1
		golapack.Zhesvrook('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesvrook('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesvrook('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhesvrook('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhesvrook('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.Zhesvrook('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZHESV_ROOK", &info, lerr, ok, t)

	} else if string(c2) == "HK" {
		//        ZSYSV_RK
		//
		//        Test error exits of the driver that uses factorization
		//        of a Hermitian indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "ZHESV_RK"
		*infot = 1
		golapack.Zhesvrk('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesvrk('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesvrk('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhesvrk('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhesvrk('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhesvrk('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhesvrk('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZHESV_RK", &info, lerr, ok, t)

	} else if string(c2) == "HA" {
		//        ZHESV_AA
		*srnamt = "ZHESV_AA"
		*infot = 1
		golapack.Zhesvaa('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesvaa('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesvaa('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA", &info, lerr, ok, t)
		*infot = 8
		golapack.Zhesvaa('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA", &info, lerr, ok, t)

	} else if string(c2) == "H2" {
		//        ZHESV_AASEN_2STAGE
		*srnamt = "ZHESV_AA_2STAGE"
		*infot = 1
		golapack.Zhesvaa2stage('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhesvaa2stage('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhesvaa2stage('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zhesvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhesvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhesvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHESV_AA_2STAGE", &info, lerr, ok, t)

	} else if string(c2) == "S2" {
		//        ZSYSV_AASEN_2STAGE
		*srnamt = "ZSYSV_AA_2STAGE"
		*infot = 1
		golapack.Zsysvaa2stage('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsysvaa2stage('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.Zsysvaa2stage('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.Zsysvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.Zsysvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 8; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.Zsysvaa2stage('U', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), a.CVector(0, 0), func() *int { y := 1; return &y }(), &ip, &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_AA_2STAGE", &info, lerr, ok, t)

	} else if string(c2) == "HP" {
		//        ZHPSV
		*srnamt = "ZHPSV "
		*infot = 1
		golapack.Zhpsv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhpsv('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhpsv('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zhpsv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZHPSV ", &info, lerr, ok, t)

		//        ZHPSVX
		*srnamt = "ZHPSVX"
		*infot = 1
		golapack.Zhpsvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zhpsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zhpsvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zhpsvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zhpsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zhpsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZHPSVX", &info, lerr, ok, t)

	} else if string(c2) == "SY" {
		//        ZSYSV
		*srnamt = "ZSYSV "
		*infot = 1
		golapack.Zsysv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsysv('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zsysv('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Zsysv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zsysv('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Zsysv('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZSYSV ", &info, lerr, ok, t)

		//        ZSYSVX
		*srnamt = "ZSYSVX"
		*infot = 1
		golapack.Zsysvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsysvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zsysvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zsysvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Zsysvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Zsysvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zsysvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Zsysvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Zsysvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, func() *int { y := 3; return &y }(), rw, &info)
		Chkxer("ZSYSVX", &info, lerr, ok, t)

	} else if string(c2) == "SR" {
		//        ZSYSV_ROOK
		*srnamt = "ZSYSV_ROOK"
		*infot = 1
		golapack.Zsysvrook('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsysvrook('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_ROOK", &info, lerr, ok, t)
		*infot = 3
		golapack.Zsysvrook('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_ROOK", &info, lerr, ok, t)
		*infot = 8
		golapack.Zsysvrook('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.Zsysvrook('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYSV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.Zsysvrook('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)

	} else if string(c2) == "SK" {
		//        ZSYSV_RK
		//
		//        Test error exits of the driver that uses factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "ZSYSV_RK"
		*infot = 1
		golapack.Zsysvrk('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 2
		golapack.Zsysvrk('U', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 3
		golapack.Zsysvrk('U', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 5
		golapack.Zsysvrk('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 9
		golapack.Zsysvrk('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.Zsysvrk('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, func() *int { y := 0; return &y }(), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.Zsysvrk('U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), e, &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), w, toPtr(-2), &info)
		Chkxer("ZSYSV_RK", &info, lerr, ok, t)

	} else if string(c2) == "SP" {
		//        ZSPSV
		*srnamt = "ZSPSV "
		*infot = 1
		golapack.Zspsv('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Zspsv('U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Zspsv('U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Zspsv('U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZSPSV ", &info, lerr, ok, t)

		//        ZSPSVX
		*srnamt = "ZSPSVX"
		*infot = 1
		golapack.Zspsvx('/', 'U', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Zspsvx('N', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Zspsvx('N', 'U', toPtr(-1), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Zspsvx('N', 'U', func() *int { y := 0; return &y }(), toPtr(-1), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Zspsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Zspsvx('N', 'U', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), &rcond, r1, r2, w, rw, &info)
		Chkxer("ZSPSVX", &info, lerr, ok, t)
	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s drivers passed the tests of the error exits\n", path)
	} else {
		fmt.Printf(" *** %3s drivers failed the tests of the error exits ***\n", path)
	}
}
