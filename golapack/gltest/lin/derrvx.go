package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrvx tests the error exits for the DOUBLE PRECISION driver routines
// for solving linear systems of equations.
func Derrvx(path []byte, t *testing.T) {
	var eq byte
	var rcond float64
	var i, info, j, nmax int
	ip := make([]int, 4)
	iw := make([]int, 4)
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4

	a := mf(4, 4, opts)
	af := mf(4, 4, opts)
	ap := vf(4 * 4)
	afp := vf(4 * 4)
	b := mf(4, 1, opts)
	c := vf(4)
	e := vf(4)
	r := vf(4)
	w := vf(4)
	x := mf(4, 1, opts)
	r1 := vf(4)
	r2 := vf(4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1/float64(i+j))
			af.Set(i-1, j-1, 1/float64(i+j))
		}
		b.SetIdx(j-1, 0)
		e.Set(j-1, 0)
		r1.Set(j-1, 0)
		r2.Set(j-1, 0)
		w.Set(j-1, 0)
		x.SetIdx(j-1, 0)
		c.Set(j-1, 0)
		r.Set(j-1, 0)
		ip[j-1] = j
	}
	eq = ' '
	*ok = true

	if string(c2) == "GE" {
		//        DGESV
		*srnamt = "DGESV "
		*infot = 1
		golapack.Dgesv(toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGESV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesv(toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGESV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgesv(toPtr(2), toPtr(1), a, toPtr(1), &ip, b, toPtr(2), &info)
		Chkxer("DGESV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgesv(toPtr(2), toPtr(1), a, toPtr(2), &ip, b, toPtr(1), &info)
		Chkxer("DGESV ", &info, lerr, ok, t)

		//        DGESVX
		*srnamt = "DGESVX"
		*infot = 1
		golapack.Dgesvx('/', 'N', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesvx('N', '/', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgesvx('N', 'N', toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgesvx('N', 'N', toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgesvx('N', 'N', toPtr(2), toPtr(1), a, toPtr(1), af, toPtr(2), &ip, &eq, r, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgesvx('N', 'N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(1), &ip, &eq, r, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 10
		eq = '/'
		golapack.Dgesvx('F', 'N', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 11
		eq = 'R'
		golapack.Dgesvx('F', 'N', toPtr(1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 12
		eq = 'C'
		golapack.Dgesvx('F', 'N', toPtr(1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dgesvx('N', 'N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(2), &ip, &eq, r, c, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dgesvx('N', 'N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(2), &ip, &eq, r, c, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGESVX", &info, lerr, ok, t)

	} else if string(c2) == "GB" {
		//        DGBSV
		*srnamt = "DGBSV "
		*infot = 1
		golapack.Dgbsv(toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbsv(toPtr(1), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbsv(toPtr(1), toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbsv(toPtr(0), toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbsv(toPtr(1), toPtr(1), toPtr(1), toPtr(0), a, toPtr(3), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgbsv(toPtr(2), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBSV ", &info, lerr, ok, t)

		//        DGBSVX
		*srnamt = "DGBSVX"
		*infot = 1
		golapack.Dgbsvx('/', 'N', toPtr(0), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbsvx('N', '/', toPtr(0), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbsvx('N', 'N', toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbsvx('N', 'N', toPtr(1), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgbsvx('N', 'N', toPtr(1), toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbsvx('N', 'N', toPtr(0), toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgbsvx('N', 'N', toPtr(1), toPtr(1), toPtr(1), toPtr(0), a, toPtr(2), af, toPtr(4), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgbsvx('N', 'N', toPtr(1), toPtr(1), toPtr(1), toPtr(0), a, toPtr(3), af, toPtr(3), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 12
		eq = '/'
		golapack.Dgbsvx('F', 'N', toPtr(0), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 13
		eq = 'R'
		golapack.Dgbsvx('F', 'N', toPtr(1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 14
		eq = 'C'
		golapack.Dgbsvx('F', 'N', toPtr(1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dgbsvx('N', 'N', toPtr(2), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Dgbsvx('N', 'N', toPtr(2), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, &eq, r, c, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGBSVX", &info, lerr, ok, t)

	} else if string(c2) == "GT" {
		//        DGTSV
		*srnamt = "DGTSV "
		*infot = 1
		golapack.Dgtsv(toPtr(-1), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), b, toPtr(1), &info)
		Chkxer("DGTSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgtsv(toPtr(0), toPtr(-1), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), b, toPtr(1), &info)
		Chkxer("DGTSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgtsv(toPtr(2), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), b, toPtr(1), &info)
		Chkxer("DGTSV ", &info, lerr, ok, t)

		//        DGTSVX
		*srnamt = "DGTSVX"
		*infot = 1
		golapack.Dgtsvx('/', 'N', toPtr(0), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgtsvx('N', '/', toPtr(0), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgtsvx('N', 'N', toPtr(-1), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgtsvx('N', 'N', toPtr(0), toPtr(-1), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dgtsvx('N', 'N', toPtr(2), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dgtsvx('N', 'N', toPtr(2), toPtr(0), a.VectorIdx(0), a.VectorIdx(1), a.VectorIdx(2), af.VectorIdx(0), af.VectorIdx(1), af.VectorIdx(2), af.VectorIdx(3), &ip, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DGTSVX", &info, lerr, ok, t)

	} else if string(c2) == "PO" {
		//        DPOSV
		*srnamt = "DPOSV "
		*infot = 1
		golapack.Dposv('/', toPtr(0), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPOSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dposv('U', toPtr(-1), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPOSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dposv('U', toPtr(0), toPtr(-1), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPOSV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dposv('U', toPtr(2), toPtr(0), a, toPtr(1), b, toPtr(2), &info)
		Chkxer("DPOSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dposv('U', toPtr(2), toPtr(0), a, toPtr(2), b, toPtr(1), &info)
		Chkxer("DPOSV ", &info, lerr, ok, t)

		//        DPOSVX
		*srnamt = "DPOSVX"
		*infot = 1
		golapack.Dposvx('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dposvx('N', '/', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dposvx('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dposvx('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dposvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(1), af, toPtr(2), &eq, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dposvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(1), &eq, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 9
		eq = '/'
		golapack.Dposvx('F', 'U', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 10
		eq = 'Y'
		golapack.Dposvx('F', 'U', toPtr(1), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Dposvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(2), &eq, c, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)
		*infot = 14
		golapack.Dposvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(2), &eq, c, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPOSVX", &info, lerr, ok, t)

	} else if string(c2) == "PP" {
		//        DPPSV
		*srnamt = "DPPSV "
		*infot = 1
		golapack.Dppsv('/', toPtr(0), toPtr(0), ap, b, toPtr(1), &info)
		Chkxer("DPPSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dppsv('U', toPtr(-1), toPtr(0), ap, b, toPtr(1), &info)
		Chkxer("DPPSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dppsv('U', toPtr(0), toPtr(-1), ap, b, toPtr(1), &info)
		Chkxer("DPPSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dppsv('U', toPtr(2), toPtr(0), ap, b, toPtr(1), &info)
		Chkxer("DPPSV ", &info, lerr, ok, t)

		//        DPPSVX
		*srnamt = "DPPSVX"
		*infot = 1
		golapack.Dppsvx('/', 'U', toPtr(0), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dppsvx('N', '/', toPtr(0), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dppsvx('N', 'U', toPtr(-1), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dppsvx('N', 'U', toPtr(0), toPtr(-1), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 7
		eq = '/'
		golapack.Dppsvx('F', 'U', toPtr(0), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 8
		eq = 'Y'
		golapack.Dppsvx('F', 'U', toPtr(1), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dppsvx('N', 'U', toPtr(2), toPtr(0), ap, afp, &eq, c, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)
		*infot = 12
		golapack.Dppsvx('N', 'U', toPtr(2), toPtr(0), ap, afp, &eq, c, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPPSVX", &info, lerr, ok, t)

	} else if string(c2) == "PB" {
		//        DPBSV
		*srnamt = "DPBSV "
		*infot = 1
		golapack.Dpbsv('/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbsv('U', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbsv('U', toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpbsv('U', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dpbsv('U', toPtr(1), toPtr(1), toPtr(0), a, toPtr(1), b, toPtr(2), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)
		*infot = 8
		golapack.Dpbsv('U', toPtr(2), toPtr(0), toPtr(0), a, toPtr(1), b, toPtr(1), &info)
		Chkxer("DPBSV ", &info, lerr, ok, t)

		//        DPBSVX
		*srnamt = "DPBSVX"
		*infot = 1
		golapack.Dpbsvx('/', 'U', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dpbsvx('N', '/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dpbsvx('N', 'U', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dpbsvx('N', 'U', toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dpbsvx('N', 'U', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dpbsvx('N', 'U', toPtr(1), toPtr(1), toPtr(0), a, toPtr(1), af, toPtr(2), &eq, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dpbsvx('N', 'U', toPtr(1), toPtr(1), toPtr(0), a, toPtr(2), af, toPtr(1), &eq, c, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 10
		eq = '/'
		golapack.Dpbsvx('F', 'U', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 11
		eq = 'Y'
		golapack.Dpbsvx('F', 'U', toPtr(1), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Dpbsvx('N', 'U', toPtr(2), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)
		*infot = 15
		golapack.Dpbsvx('N', 'U', toPtr(2), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &eq, c, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DPBSVX", &info, lerr, ok, t)

	} else if string(c2) == "PT" {
		//        DPTSV
		*srnamt = "DPTSV "
		*infot = 1
		golapack.Dptsv(toPtr(-1), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), b, toPtr(1), &info)
		Chkxer("DPTSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dptsv(toPtr(0), toPtr(-1), a.Vector(0, 0), a.Vector(0, 1), b, toPtr(1), &info)
		Chkxer("DPTSV ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dptsv(toPtr(2), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), b, toPtr(1), &info)
		Chkxer("DPTSV ", &info, lerr, ok, t)

		//        DPTSVX
		*srnamt = "DPTSVX"
		*infot = 1
		golapack.Dptsvx('/', toPtr(0), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), af.Vector(0, 0), af.Vector(0, 1), b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &info)
		Chkxer("DPTSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dptsvx('N', toPtr(-1), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), af.Vector(0, 0), af.Vector(0, 1), b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &info)
		Chkxer("DPTSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dptsvx('N', toPtr(0), toPtr(-1), a.Vector(0, 0), a.Vector(0, 1), af.Vector(0, 0), af.Vector(0, 1), b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &info)
		Chkxer("DPTSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dptsvx('N', toPtr(2), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), af.Vector(0, 0), af.Vector(0, 1), b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &info)
		Chkxer("DPTSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dptsvx('N', toPtr(2), toPtr(0), a.Vector(0, 0), a.Vector(0, 1), af.Vector(0, 0), af.Vector(0, 1), b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &info)
		Chkxer("DPTSVX", &info, lerr, ok, t)

	} else if string(c2) == "SY" {
		//        DSYSV
		*srnamt = "DSYSV "
		*infot = 1
		golapack.Dsysv('/', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsysv('U', toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsysv('U', toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dsysv('U', toPtr(2), toPtr(0), a, toPtr(1), &ip, b, toPtr(2), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsysv('U', toPtr(2), toPtr(0), a, toPtr(2), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsysv('U', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)
		*infot = 10
		golapack.Dsysv('U', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(-2), &info)
		Chkxer("DSYSV ", &info, lerr, ok, t)

		//        DSYSVX
		*srnamt = "DSYSVX"
		*infot = 1
		golapack.Dsysvx('/', 'U', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, toPtr(1), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dsysvx('N', '/', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, toPtr(1), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dsysvx('N', 'U', toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, toPtr(1), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dsysvx('N', 'U', toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, toPtr(1), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 6
		golapack.Dsysvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(1), af, toPtr(2), &ip, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, toPtr(4), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dsysvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(1), &ip, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, toPtr(4), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dsysvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(2), &ip, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, toPtr(4), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Dsysvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(2), &ip, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, toPtr(4), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)
		*infot = 18
		golapack.Dsysvx('N', 'U', toPtr(2), toPtr(0), a, toPtr(2), af, toPtr(2), &ip, b, toPtr(2), x, toPtr(2), &rcond, r1, r2, w, toPtr(3), &iw, &info)
		Chkxer("DSYSVX", &info, lerr, ok, t)

	} else if string(c2) == "SR" {
		//        DSYSV_ROOK
		*srnamt = "DSYSV_ROOK"
		*infot = 1
		golapack.DsysvRook('/', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsysvRook('U', toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 3
		golapack.DsysvRook('U', toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 5
		golapack.DsysvRook('U', toPtr(2), toPtr(0), a, toPtr(1), &ip, b, toPtr(2), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 8
		golapack.DsysvRook('U', toPtr(2), toPtr(0), a, toPtr(2), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.DsysvRook('U', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)
		*infot = 10
		golapack.DsysvRook('U', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(-2), &info)
		Chkxer("DSYSV_ROOK", &info, lerr, ok, t)

	} else if string(c2) == "SK" {
		//        DSYSV_RK
		//
		//        Test error exits of the driver that uses factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "DSYSV_RK"
		*infot = 1
		golapack.DsysvRk('/', toPtr(0), toPtr(0), a, toPtr(1), e, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 2
		golapack.DsysvRk('U', toPtr(-1), toPtr(0), a, toPtr(1), e, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 3
		golapack.DsysvRk('U', toPtr(0), toPtr(-1), a, toPtr(1), e, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 5
		golapack.DsysvRk('U', toPtr(2), toPtr(0), a, toPtr(1), e, &ip, b, toPtr(2), w, toPtr(1), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 9
		golapack.DsysvRk('U', toPtr(2), toPtr(0), a, toPtr(2), e, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.DsysvRk('U', toPtr(0), toPtr(0), a, toPtr(1), e, &ip, b, toPtr(1), w, toPtr(0), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)
		*infot = 11
		golapack.DsysvRk('U', toPtr(0), toPtr(0), a, toPtr(1), e, &ip, b, toPtr(1), w, toPtr(-2), &info)
		Chkxer("DSYSV_RK", &info, lerr, ok, t)

	} else if string(c2) == "SA" {
		//        DSYSV_AA
		*srnamt = "DSYSV_AA"
		*infot = 1
		golapack.DsysvAa('/', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA", &info, lerr, ok, t)
		*infot = 2
		golapack.DsysvAa('U', toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA", &info, lerr, ok, t)
		*infot = 3
		golapack.DsysvAa('U', toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA", &info, lerr, ok, t)
		*infot = 8
		golapack.DsysvAa('U', toPtr(2), toPtr(0), a, toPtr(2), &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA", &info, lerr, ok, t)

	} else if string(c2) == "S2" {
		//        DSYSV_AASEN_2STAGE
		*srnamt = "DSYSV_AA_2STAGE"
		*infot = 1
		golapack.DsysvAa2stage('/', toPtr(0), toPtr(0), a, toPtr(1), a, toPtr(1), &ip, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 2
		golapack.DsysvAa2stage('U', toPtr(-1), toPtr(0), a, toPtr(1), a, toPtr(1), &ip, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 3
		golapack.DsysvAa2stage('U', toPtr(0), toPtr(-1), a, toPtr(1), a, toPtr(1), &ip, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 5
		golapack.DsysvAa2stage('U', toPtr(2), toPtr(1), a, toPtr(1), a, toPtr(1), &ip, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 11
		golapack.DsysvAa2stage('U', toPtr(2), toPtr(1), a, toPtr(2), a, toPtr(8), &ip, &ip, b, toPtr(1), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)
		*infot = 7
		golapack.DsysvAa2stage('U', toPtr(2), toPtr(1), a, toPtr(2), a, toPtr(1), &ip, &ip, b, toPtr(2), w, toPtr(1), &info)
		Chkxer("DSYSV_AA_2STAGE", &info, lerr, ok, t)

	} else if string(c2) == "SP" {
		//        DSPSV
		*srnamt = "DSPSV "
		*infot = 1
		golapack.Dspsv('/', toPtr(0), toPtr(0), ap, &ip, b, toPtr(1), &info)
		Chkxer("DSPSV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspsv('U', toPtr(-1), toPtr(0), ap, &ip, b, toPtr(1), &info)
		Chkxer("DSPSV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dspsv('U', toPtr(0), toPtr(-1), ap, &ip, b, toPtr(1), &info)
		Chkxer("DSPSV ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dspsv('U', toPtr(2), toPtr(0), ap, &ip, b, toPtr(1), &info)
		Chkxer("DSPSV ", &info, lerr, ok, t)

		//        DSPSVX
		*srnamt = "DSPSVX"
		*infot = 1
		golapack.Dspsvx('/', 'U', toPtr(0), toPtr(0), ap, afp, &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dspsvx('N', '/', toPtr(0), toPtr(0), ap, afp, &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dspsvx('N', 'U', toPtr(-1), toPtr(0), ap, afp, &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dspsvx('N', 'U', toPtr(0), toPtr(-1), ap, afp, &ip, b, toPtr(1), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dspsvx('N', 'U', toPtr(2), toPtr(0), ap, afp, &ip, b, toPtr(1), x, toPtr(2), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dspsvx('N', 'U', toPtr(2), toPtr(0), ap, afp, &ip, b, toPtr(2), x, toPtr(1), &rcond, r1, r2, w, &iw, &info)
		Chkxer("DSPSVX", &info, lerr, ok, t)

	}

	//     Print a summary line.
	if *ok {
		fmt.Printf(" %3s drivers passed the tests of the error exits\n", path)
	} else {
		fmt.Printf(" *** %3s drivers failed the tests of the error exits ***\n", path)
	}
}
