package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrvx tests the error exits for the DOUBLE PRECISION driver routines
// for solving linear systems of equations.
func derrvx(path string, t *testing.T) {
	var eq byte
	var i, j, nmax int
	var err error
	ip := make([]int, 4)
	iw := make([]int, 4)
	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
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

	if c2 == "ge" {
		//        DGESV
		*srnamt = "Dgesv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgesv(-1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgesv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dgesv(0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgesv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dgesv(2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("Dgesv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dgesv(2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgesv", err)

		//        Dgesvx
		*srnamt = "Dgesvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Dgesvx('/', NoTrans, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, _, _, err = golapack.Dgesvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equedOut == 'N'): fact='F', equed='/'")
		eq = '/'
		_, _, _, err = golapack.Dgesvx('F', NoTrans, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'R'
		_, _, _, err = golapack.Dgesvx('F', NoTrans, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'C'
		_, _, _, err = golapack.Dgesvx('F', NoTrans, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Dgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgesvx", err)

	} else if c2 == "gb" {
		//        DGBSV
		*srnamt = "Dgbsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgbsv(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Dgbsv(1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Dgbsv(1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dgbsv(0, 0, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)
		*errt = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Dgbsv(1, 1, 1, 0, a.Off(0, 0).UpdateRows(3), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)
		*errt = fmt.Errorf("b.Rows < max(n, 1): b.Rows=1, n=2")
		_, err = golapack.Dgbsv(2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgbsv", err)

		//        Dgbsvx
		*srnamt = "Dgbsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		eq, _, _, err = golapack.Dgbsvx('/', NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		eq, _, _, err = golapack.Dgbsvx('N', '/', 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, -1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 0, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(4), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("afb.Rows < 2*kl+ku+1: afb.Rows=3, kl=1, ku=1")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(3), af.Off(0, 0).UpdateRows(3), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equed == 'N'): fact='F', equed='/'")
		eq = '/'
		eq, _, _, err = golapack.Dgbsvx('F', NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("rcmin <= zero: equed='R', rcmin=0")
		eq = 'R'
		eq, _, _, err = golapack.Dgbsvx('F', NoTrans, 1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("rcmin <= zero: equed='C', rcmin=0")
		eq = 'C'
		eq, _, _, err = golapack.Dgbsvx('F', NoTrans, 1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		eq, _, _, err = golapack.Dgbsvx('N', NoTrans, 2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgbsvx", err)

	} else if c2 == "gt" {
		//        DGTSV
		*srnamt = "Dgtsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgtsv(-1, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgtsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dgtsv(0, -1, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgtsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dgtsv(2, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dgtsv", err)

		//        Dgtsvx
		*srnamt = "Dgtsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Dgtsvx('/', NoTrans, 0, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)
		*errt = fmt.Errorf("!trans.IsValid(): trans=Unrecognized: /")
		_, _, err = golapack.Dgtsvx('N', '/', 0, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgtsvx('N', NoTrans, -1, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dgtsvx('N', NoTrans, 0, -1, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Dgtsvx('N', NoTrans, 2, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Dgtsvx('N', NoTrans, 2, 0, a.OffIdx(0).Vector(), a.OffIdx(1).Vector(), a.OffIdx(2).Vector(), af.OffIdx(0).Vector(), af.OffIdx(1).Vector(), af.OffIdx(2).Vector(), af.OffIdx(3).Vector(), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dgtsvx", err)

	} else if c2 == "po" {
		//        Dposv
		*srnamt = "Dposv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dposv('/', 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dposv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dposv(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dposv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dposv(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dposv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dposv(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2))
		chkxer2("Dposv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dposv(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dposv", err)

		//        Dposvx
		*srnamt = "Dposvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Dposvx('/', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Dposvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dposvx('N', Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Dposvx('N', Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Dposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, _, err = golapack.Dposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='F', equed='/'")
		eq = '/'
		_, _, _, err = golapack.Dposvx('F', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		_, _, _, err = golapack.Dposvx('F', Upper, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Dposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dposvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Dposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dposvx", err)

	} else if c2 == "pp" {
		//        DPPSV
		*srnamt = "Dppsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dppsv('/', 0, 0, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dppsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dppsv(Upper, -1, 0, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dppsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dppsv(Upper, 0, -1, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dppsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dppsv(Upper, 2, 0, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dppsv", err)

		//        Dppsvx
		*srnamt = "Dppsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		eq, _, _, err = golapack.Dppsvx('/', Upper, 0, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		eq, _, _, err = golapack.Dppsvx('N', '/', 0, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		eq, _, _, err = golapack.Dppsvx('N', Upper, -1, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		eq, _, _, err = golapack.Dppsvx('N', Upper, 0, -1, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='F', equed='/'")
		eq = '/'
		eq, _, _, err = golapack.Dppsvx('F', Upper, 0, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		eq, _, _, err = golapack.Dppsvx('F', Upper, 1, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		eq, _, _, err = golapack.Dppsvx('N', Upper, 2, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		eq, _, _, err = golapack.Dppsvx('N', Upper, 2, 0, ap, afp, eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dppsvx", err)

	} else if c2 == "pb" {
		//        DPBSV
		*srnamt = "Dpbsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpbsv('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpbsv(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbsv", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dpbsv(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dpbsv(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbsv", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dpbsv(Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2))
		chkxer2("Dpbsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dpbsv(Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbsv", err)

		//        Dpbsvx
		*srnamt = "Dpbsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Dpbsvx('/', Upper, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Dpbsvx('N', '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dpbsvx('N', Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("afb.Rows < kd+1: afb.Rows=1, kd=1")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equedOut == 'N'): fact='F', equed='/'")
		eq = '/'
		_, _, _, err = golapack.Dpbsvx('F', Upper, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		_, _, _, err = golapack.Dpbsvx('F', Upper, 1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Dpbsvx('N', Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbsvx", err)

	} else if c2 == "pt" {
		//        DPTSV
		*srnamt = "Dptsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dptsv(-1, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dptsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dptsv(0, -1, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dptsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dptsv(2, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dptsv", err)

		//        Dptsvx
		*srnamt = "Dptsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Dptsvx('/', 0, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), af.Off(0, 0).Vector(), af.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dptsvx('N', -1, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), af.Off(0, 0).Vector(), af.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dptsvx('N', 0, -1, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), af.Off(0, 0).Vector(), af.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Dptsvx('N', 2, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), af.Off(0, 0).Vector(), af.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w)
		chkxer2("Dptsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Dptsvx('N', 2, 0, a.Off(0, 0).Vector(), a.Off(0, 1).Vector(), af.Off(0, 0).Vector(), af.Off(0, 1).Vector(), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w)
		chkxer2("Dptsvx", err)

	} else if c2 == "sy" {
		//        DSYSV
		*srnamt = "Dsysv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsysv('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsysv(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dsysv(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsysv(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dsysv(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Dsysv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 0)
		chkxer2("Dsysv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Dsysv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, -2)
		chkxer2("Dsysv", err)

		//        Dsysvx
		*srnamt = "Dsysvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Dsysvx('/', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, 1, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Dsysvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, 1, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dsysvx('N', Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, 1, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dsysvx('N', Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, 1, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Dsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, 4, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, err = golapack.Dsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, 4, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Dsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, 4, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Dsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, 4, &iw)
		chkxer2("Dsysvx", err)
		*errt = fmt.Errorf("lwork < max(1, 3*n) && !lquery: lwork=3, n=2, lquery=false")
		_, _, err = golapack.Dsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, 3, &iw)
		chkxer2("Dsysvx", err)

	} else if c2 == "sr" {
		//        DsysvRook
		*srnamt = "DsysvRook"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsysvRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsysvRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsysvRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsysvRook(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.DsysvRook(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.DsysvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 0)
		chkxer2("DsysvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.DsysvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, -2)
		chkxer2("DsysvRook", err)

	} else if c2 == "sk" {
		//        DsysvRk
		//
		//        Test error exits of the driver that uses factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "DsysvRk"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsysvRk('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsysvRk(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsysvRk(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsysvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.DsysvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), e, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.DsysvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1), w, 0)
		chkxer2("DsysvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.DsysvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1), w, -2)
		chkxer2("DsysvRk", err)

	} else if c2 == "sa" {
		//        DsysvAa
		*srnamt = "DsysvAa"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsysvAa('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsysvAa(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsysvAa(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa", err)
		// *errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		// *errt = fmt.Errorf("lwork < max(2*n, 3*n-2) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		_, err = golapack.DsysvAa(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa", err)

	} else if c2 == "s2" {
		//        DsysvAa2stage
		*srnamt = "DsysvAa2stage"
		// *errt = fmt.Errorf("lwork < n && !wquery: lwork=%v, n=%v, wquery=%v", lwork, n, wquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsysvAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), a, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsysvAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), a, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsysvAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), a, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), a, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.DsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a, 8, &ip, &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsysvAa2stage", err)
		*errt = fmt.Errorf("ltb < (4*n) && !tquery: ltb=1, n=2, tquery=false")
		_, err = golapack.DsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a, 1, &ip, &ip, b.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("DsysvAa2stage", err)

	} else if c2 == "sp" {
		//        DSPSV
		*srnamt = "Dspsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dspsv('/', 0, 0, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dspsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dspsv(Upper, -1, 0, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dspsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dspsv(Upper, 0, -1, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dspsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dspsv(Upper, 2, 0, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dspsv", err)

		//        Dspsvx
		*srnamt = "Dspsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Dspsvx('/', Upper, 0, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Dspsvx('N', '/', 0, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dspsvx('N', Upper, -1, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Dspsvx('N', Upper, 0, -1, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Dspsvx('N', Upper, 2, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Dspsvx('N', Upper, 2, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dspsvx", err)

	}

	//     Print a summary line.
	// if *ok {
	// 	fmt.Printf(" %3s drivers passed the tests of the error exits\n", path)
	// } else {
	// 	fmt.Printf(" *** %3s drivers failed the tests of the error exits ***\n", path)
	// }

	if !(*ok) {
		t.Fail()
	}
}
