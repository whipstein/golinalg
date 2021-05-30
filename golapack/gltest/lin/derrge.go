package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrge tests the error exits for the DOUBLE PRECISION routines
// for general matrices.
func Derrge(path []byte, t *testing.T) {
	var anrm, ccond, rcond float64
	var i, info, j, lw, nmax int
	ip := make([]int, 4)
	iw := make([]int, 4)
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4
	lw = 3 * nmax

	a := mf(4, 4, opts)
	af := mf(4, 4, opts)
	b := mf(4, 1, opts)
	w := mf(lw, 1, opts)
	x := mf(4, 1, opts)
	r1 := vf(4)
	r2 := vf(4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.SetIdx(j-1, 0.)
		x.SetIdx(j-1, 0.)
		ip[j-1] = j
		iw[j-1] = j
	}
	*lerr = false
	*ok = true

	if string(c2) == "GE" {
		//        Test error exits of the routines that use the LU decomposition
		//        of a general matrix.
		//
		//        DGETRF
		*srnamt = "DGETRF"
		*infot = 1
		golapack.Dgetrf(toPtr(-1), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGETRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgetrf(toPtr(0), toPtr(-1), a, toPtr(1), &ip, &info)
		Chkxer("DGETRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgetrf(toPtr(2), toPtr(1), a, toPtr(1), &ip, &info)
		Chkxer("DGETRF", &info, lerr, ok, t)

		//        DGETF2
		*srnamt = "DGETF2"
		*infot = 1
		golapack.Dgetf2(toPtr(-1), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGETF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgetf2(toPtr(0), toPtr(-1), a, toPtr(1), &ip, &info)
		Chkxer("DGETF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgetf2(toPtr(2), toPtr(1), a, toPtr(1), &ip, &info)
		Chkxer("DGETF2", &info, lerr, ok, t)

		//        DGETRI
		*srnamt = "DGETRI"
		*infot = 1
		golapack.Dgetri(toPtr(-1), a, toPtr(1), &ip, w, &lw, &info)
		Chkxer("DGETRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgetri(toPtr(2), a, toPtr(1), &ip, w, &lw, &info)
		Chkxer("DGETRI", &info, lerr, ok, t)

		//        DGETRS
		*srnamt = "DGETRS"
		*infot = 1
		golapack.Dgetrs('/', toPtr(0), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGETRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgetrs('N', toPtr(-1), toPtr(0), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGETRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgetrs('N', toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGETRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgetrs('N', toPtr(2), toPtr(1), a, toPtr(1), &ip, b, toPtr(2), &info)
		Chkxer("DGETRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgetrs('N', toPtr(2), toPtr(1), a, toPtr(2), &ip, b, toPtr(1), &info)
		Chkxer("DGETRS", &info, lerr, ok, t)

		//        DGERFS
		*srnamt = "DGERFS"
		*infot = 1
		golapack.Dgerfs('/', toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgerfs('N', toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgerfs('N', toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgerfs('N', toPtr(2), toPtr(1), a, toPtr(1), af, toPtr(2), &ip, b, toPtr(2), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgerfs('N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(1), &ip, b, toPtr(2), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgerfs('N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(2), &ip, b, toPtr(1), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgerfs('N', toPtr(2), toPtr(1), a, toPtr(2), af, toPtr(2), &ip, b, toPtr(2), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGERFS", &info, lerr, ok, t)

		//        DGECON
		*srnamt = "DGECON"
		*infot = 1
		golapack.Dgecon('/', toPtr(0), a, toPtr(1), &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGECON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgecon('1', toPtr(-1), a, toPtr(1), &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGECON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgecon('1', toPtr(2), a, toPtr(1), &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGECON", &info, lerr, ok, t)

		//        DGEEQU
		*srnamt = "DGEEQU"
		*infot = 1
		golapack.Dgeequ(toPtr(-1), toPtr(0), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGEEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgeequ(toPtr(0), toPtr(-1), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGEEQU", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgeequ(toPtr(2), toPtr(2), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGEEQU", &info, lerr, ok, t)

	} else if string(c2) == "GB" {
		//        Test error exits of the routines that use the LU decomposition
		//        of a general band matrix.
		//
		//        DGBTRF
		*srnamt = "DGBTRF"
		*infot = 1
		golapack.Dgbtrf(toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbtrf(toPtr(0), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbtrf(toPtr(1), toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbtrf(toPtr(1), toPtr(1), toPtr(0), toPtr(-1), a, toPtr(1), &ip, &info)
		Chkxer("DGBTRF", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbtrf(toPtr(2), toPtr(2), toPtr(1), toPtr(1), a, toPtr(3), &ip, &info)
		Chkxer("DGBTRF", &info, lerr, ok, t)

		//        DGBTF2
		*srnamt = "DGBTF2"
		*infot = 1
		golapack.Dgbtf2(toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbtf2(toPtr(0), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTF2", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbtf2(toPtr(1), toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), &ip, &info)
		Chkxer("DGBTF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbtf2(toPtr(1), toPtr(1), toPtr(0), toPtr(-1), a, toPtr(1), &ip, &info)
		Chkxer("DGBTF2", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbtf2(toPtr(2), toPtr(2), toPtr(1), toPtr(1), a, toPtr(3), &ip, &info)
		Chkxer("DGBTF2", &info, lerr, ok, t)

		//        DGBTRS
		*srnamt = "DGBTRS"
		*infot = 1
		golapack.Dgbtrs('/', toPtr(0), toPtr(0), toPtr(0), toPtr(1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbtrs('N', toPtr(-1), toPtr(0), toPtr(0), toPtr(1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbtrs('N', toPtr(1), toPtr(-1), toPtr(0), toPtr(1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbtrs('N', toPtr(1), toPtr(0), toPtr(-1), toPtr(1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgbtrs('N', toPtr(1), toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgbtrs('N', toPtr(2), toPtr(1), toPtr(1), toPtr(1), a, toPtr(3), &ip, b, toPtr(2), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgbtrs('N', toPtr(2), toPtr(0), toPtr(0), toPtr(1), a, toPtr(1), &ip, b, toPtr(1), &info)
		Chkxer("DGBTRS", &info, lerr, ok, t)

		//        DGBRFS
		*srnamt = "DGBRFS"
		*infot = 1
		golapack.Dgbrfs('/', toPtr(0), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbrfs('N', toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbrfs('N', toPtr(1), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbrfs('N', toPtr(1), toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgbrfs('N', toPtr(1), toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgbrfs('N', toPtr(2), toPtr(1), toPtr(1), toPtr(1), a, toPtr(2), af, toPtr(4), &ip, b, toPtr(2), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgbrfs('N', toPtr(2), toPtr(1), toPtr(1), toPtr(1), a, toPtr(3), af, toPtr(3), &ip, b, toPtr(2), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgbrfs('N', toPtr(2), toPtr(0), toPtr(0), toPtr(1), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(1), x, toPtr(2), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)
		*infot = 14
		golapack.Dgbrfs('N', toPtr(2), toPtr(0), toPtr(0), toPtr(1), a, toPtr(1), af, toPtr(1), &ip, b, toPtr(2), x, toPtr(1), r1, r2, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBRFS", &info, lerr, ok, t)

		//        DGBCON
		*srnamt = "DGBCON"
		*infot = 1
		golapack.Dgbcon('/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), &ip, &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbcon('1', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), &ip, &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbcon('1', toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), &ip, &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbcon('1', toPtr(1), toPtr(0), toPtr(-1), a, toPtr(1), &ip, &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbcon('1', toPtr(2), toPtr(1), toPtr(1), a, toPtr(3), &ip, &anrm, &rcond, w.VectorIdx(0), &iw, &info)
		Chkxer("DGBCON", &info, lerr, ok, t)

		//        DGBEQU
		*srnamt = "DGBEQU"
		*infot = 1
		golapack.Dgbequ(toPtr(-1), toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGBEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgbequ(toPtr(0), toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGBEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgbequ(toPtr(1), toPtr(1), toPtr(-1), toPtr(0), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGBEQU", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgbequ(toPtr(1), toPtr(1), toPtr(0), toPtr(-1), a, toPtr(1), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGBEQU", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgbequ(toPtr(2), toPtr(2), toPtr(1), toPtr(1), a, toPtr(2), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("DGBEQU", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
