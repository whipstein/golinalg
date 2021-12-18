package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrvx tests the error exits for the COMPLEX*16 driver routines
// for solving linear systems of equations.
func zerrvx(path string, t *testing.T) {
	var eq byte
	var i, j, nmax int
	var err error

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

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
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

	if c2 == "ge" {
		//        ZGESV
		*srnamt = "Zgesv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgesv(-1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgesv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zgesv(0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgesv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zgesv(2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("Zgesv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zgesv(2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("Zgesv", err)

		//        Zgesvx
		*srnamt = "Zgesvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Zgesvx('/', NoTrans, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, _, _, err = golapack.Zgesvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equed == 'N'): fact='F', equed='/'")
		eq = '/'
		_, _, _, err = golapack.Zgesvx('F', NoTrans, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'R'
		_, _, _, err = golapack.Zgesvx('F', NoTrans, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'C'
		_, _, _, err = golapack.Zgesvx('F', NoTrans, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Zgesvx('N', NoTrans, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, eq, r, c, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgesvx", err)

	} else if c2 == "gb" {
		//        ZGBSV
		*srnamt = "Zgbsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgbsv(-1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, err = golapack.Zgbsv(1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, err = golapack.Zgbsv(1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zgbsv(0, 0, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)
		*errt = fmt.Errorf("ab.Rows < 2*kl+ku+1: ab.Rows=3, kl=1, ku=1")
		_, err = golapack.Zgbsv(1, 1, 1, 0, a.Off(0, 0).UpdateRows(3), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)
		*errt = fmt.Errorf("b.Rows < max(n, 1): b.Rows=1, n=2")
		_, err = golapack.Zgbsv(2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zgbsv", err)

		//        Zgbsvx
		*srnamt = "Zgbsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Zgbsvx('/', NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, _, _, err = golapack.Zgbsvx('N', '/', 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, -1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("kl < 0: kl=-1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 1, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("ku < 0: ku=-1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 1, 0, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 0, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=2, kl=1, ku=1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(4), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("afb.Rows < 2*kl+ku+1: afb.Rows=3, kl=1, ku=1")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 1, 1, 1, 0, a.Off(0, 0).UpdateRows(3), af.Off(0, 0).UpdateRows(3), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equed == 'N'): fact='F', equed='/'")
		eq = '/'
		_, _, _, err = golapack.Zgbsvx('F', NoTrans, 0, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'R'
		_, _, _, err = golapack.Zgbsvx('F', NoTrans, 1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("rcmin <= zero: rcmin=0")
		eq = 'C'
		_, _, _, err = golapack.Zgbsvx('F', NoTrans, 1, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Zgbsvx('N', NoTrans, 2, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, eq, r, c, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgbsvx", err)

	} else if c2 == "gt" {
		//        ZGTSV
		*srnamt = "Zgtsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgtsv(-1, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), b.CMatrix(1, opts))
		chkxer2("Zgtsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zgtsv(0, -1, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), b.CMatrix(1, opts))
		chkxer2("Zgtsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zgtsv(2, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), b.CMatrix(1, opts))
		chkxer2("Zgtsv", err)

		//        Zgtsvx
		*srnamt = "Zgtsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zgtsvx('/', NoTrans, 0, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)
		*errt = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=Unrecognized: /")
		_, _, err = golapack.Zgtsvx('N', '/', 0, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zgtsvx('N', NoTrans, -1, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zgtsvx('N', NoTrans, 0, -1, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zgtsvx('N', NoTrans, 2, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zgtsvx('N', NoTrans, 2, 0, a.Off(0, 0).CVector(), a.Off(0, 1).CVector(), a.Off(0, 2).CVector(), af.Off(0, 0).CVector(), af.Off(0, 1).CVector(), af.Off(0, 2).CVector(), af.Off(0, 3).CVector(), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zgtsvx", err)

	} else if c2 == "po" {
		//        ZPOSV
		*srnamt = "Zposv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zposv('/', 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zposv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zposv(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zposv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zposv(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zposv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zposv(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts))
		chkxer2("Zposv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zposv(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), b.CMatrix(1, opts))
		chkxer2("Zposv", err)

		//        Zposvx
		*srnamt = "Zposvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Zposvx('/', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Zposvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zposvx('N', Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Zposvx('N', Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Zposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), eq, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, _, err = golapack.Zposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='F', equed='/', rcequ=false")
		eq = '/'
		_, _, _, err = golapack.Zposvx('F', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		_, _, _, err = golapack.Zposvx('F', Upper, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Zposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), eq, c, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Zposvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), eq, c, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zposvx", err)

	} else if c2 == "pp" {
		//        ZPPSV
		*srnamt = "Zppsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zppsv('/', 0, 0, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zppsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zppsv(Upper, -1, 0, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zppsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zppsv(Upper, 0, -1, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zppsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zppsv(Upper, 2, 0, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zppsv", err)

		//        Zppsvx
		*srnamt = "Zppsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Zppsvx('/', Upper, 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Zppsvx('N', '/', 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zppsvx('N', Upper, -1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Zppsvx('N', Upper, 0, -1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='F', equed='/', rcequ=false")
		eq = '/'
		_, _, _, err = golapack.Zppsvx('F', Upper, 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		_, _, _, err = golapack.Zppsvx('F', Upper, 1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Zppsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Zppsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), eq, c, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zppsvx", err)

	} else if c2 == "pb" {
		//        ZPBSV
		*srnamt = "Zpbsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpbsv('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpbsv(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbsv", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Zpbsv(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zpbsv(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbsv", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Zpbsv(Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts))
		chkxer2("Zpbsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zpbsv(Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbsv", err)

		//        Zpbsvx
		*srnamt = "Zpbsvx"
		*errt = fmt.Errorf("!nofact && !equil && fact != 'F': fact='/'")
		_, _, _, err = golapack.Zpbsvx('/', Upper, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Zpbsvx('N', '/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zpbsvx('N', Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), eq, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("afb.Rows < kd+1: afb.Rows=1, kd=1")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 1, 1, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='F', equed='/', rcequ=false")
		eq = '/'
		_, _, _, err = golapack.Zpbsvx('F', Upper, 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("smin <= zero: smin=0")
		eq = 'Y'
		_, _, _, err = golapack.Zpbsvx('F', Upper, 1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, _, err = golapack.Zpbsvx('N', Upper, 2, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), eq, c, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zpbsvx", err)

	} else if c2 == "pt" {
		//        ZPTSV
		*srnamt = "Zptsv"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zptsv(-1, 0, r, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zptsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zptsv(0, -1, r, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zptsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zptsv(2, 0, r, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zptsv", err)

		//        Zptsvx
		*srnamt = "Zptsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zptsvx('/', 0, 0, r, a.Off(0, 0).CVector(), rf, af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zptsvx('N', -1, 0, r, a.Off(0, 0).CVector(), rf, af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zptsvx('N', 0, -1, r, a.Off(0, 0).CVector(), rf, af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zptsvx('N', 2, 0, r, a.Off(0, 0).CVector(), rf, af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zptsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zptsvx('N', 2, 0, r, a.Off(0, 0).CVector(), rf, af.Off(0, 0).CVector(), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zptsvx", err)

	} else if c2 == "he" {
		//        ZHESV
		*srnamt = "Zhesv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhesv('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhesv(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zhesv(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhesv(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zhesv(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Zhesv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("Zhesv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Zhesv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("Zhesv", err)

		//        Zhesvx
		*srnamt = "Zhesvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zhesvx('/', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Zhesvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zhesvx('N', Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zhesvx('N', Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Zhesvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, err = golapack.Zhesvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zhesvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zhesvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, 4, rw)
		chkxer2("Zhesvx", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=3, n=2, lquery=false")
		_, _, err = golapack.Zhesvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 3, rw)
		chkxer2("Zhesvx", err)

	} else if c2 == "hr" {
		//        ZhesvRook
		*srnamt = "ZhesvRook"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhesvRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhesvRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZhesvRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRook", err)
		// *errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v")
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZhesvRook(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZhesvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("ZhesvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZhesvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("ZhesvRook", err)

	} else if c2 == "hk" {
		//        ZsysvRk
		//
		//        Test error exits of the driver that uses factorization
		//        of a Hermitian indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "ZhesvRk"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhesvRk('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhesvRk(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZhesvRk(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhesvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZhesvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZhesvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("ZhesvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZhesvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("ZhesvRk", err)

	} else if c2 == "ha" {
		//        ZhesvAa
		*srnamt = "ZhesvAa"
		// *errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
		// *errt = fmt.Errorf("lwork < max(2*n, 3*n-2) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhesvAa('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhesvAa(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZhesvAa(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZhesvAa(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa", err)

	} else if c2 == "h2" {
		//        ZhesvAaSEN_2STAGE
		*srnamt = "ZhesvAa2stage"
		// *errt = fmt.Errorf("lwork < n && !wquery: lwork=%v, n=%v, tquery=%v", lwork, n, tquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhesvAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhesvAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZhesvAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhesvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZhesvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 8, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)
		*errt = fmt.Errorf("ltb < (4*n) && !tquery: ltb=1, n=2, tquery=false")
		_, err = golapack.ZhesvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZhesvAa2stage", err)

	} else if c2 == "s2" {
		//        ZSYSV_AASEN_2STAGE
		*srnamt = "ZsysvAa2stage"
		// *errt = fmt.Errorf("lwork < n && !wquery: lwork=%v, n=%v, wquery=%v", lwork, n, wquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsysvAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsysvAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZsysvAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 8, &ip, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)
		*errt = fmt.Errorf("ltb < (4*n) && !tquery: ltb=1, n=2, tquery=false")
		_, err = golapack.ZsysvAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZsysvAa2stage", err)

	} else if c2 == "hp" {
		//        ZHPSV
		*srnamt = "Zhpsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhpsv('/', 0, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zhpsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhpsv(Upper, -1, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zhpsv", err)
		*errt = fmt.Errorf("nrhs < 0: brhs=-1")
		_, err = golapack.Zhpsv(Upper, 0, -1, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zhpsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zhpsv(Upper, 2, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zhpsv", err)

		//        Zhpsvx
		*srnamt = "Zhpsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zhpsvx('/', Upper, 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Zhpsvx('N', '/', 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zhpsvx('N', Upper, -1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zhpsvx('N', Upper, 0, -1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zhpsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zhpsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zhpsvx", err)

	} else if c2 == "sy" {
		//        ZSYSV
		*srnamt = "Zsysv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsysv('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zsysv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsysv(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zsysv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zsysv(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zsysv", err)
		// *errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=2, n=2")
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zsysv(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("Zsysv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Zsysv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("Zsysv", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Zsysv(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("Zsysv", err)

		//        Zsysvx
		*srnamt = "Zsysvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zsysvx('/', Upper, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Zsysvx('N', '/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zsysvx('N', Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zsysvx('N', Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, 1, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Zsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, _, err = golapack.Zsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, 4, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, 4, rw)
		chkxer2("Zsysvx", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=3, n=2, lquery=false")
		_, _, err = golapack.Zsysvx('N', Upper, 2, 0, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, 3, rw)
		chkxer2("Zsysvx", err)

	} else if c2 == "sr" {
		//        ZsysvRook
		*srnamt = "ZsysvRook"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsysvRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsysvRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZsysvRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRook", err)
		// *errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZsysvRook(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZsysvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("ZsysvRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZsysvRook(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, -2)

	} else if c2 == "sk" {
		//        ZsysvRk
		//
		//        Test error exits of the driver that uses factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		*srnamt = "ZsysvRk"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsysvRk('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsysvRk(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZsysvRk(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsysvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZsysvRk(Upper, 2, 0, a.Off(0, 0).UpdateRows(2), e, &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZsysvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("ZsysvRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZsysvRk(Upper, 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("ZsysvRk", err)

	} else if c2 == "sp" {
		//        ZSPSV
		*srnamt = "Zspsv"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zspsv('/', 0, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zspsv", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zspsv(Upper, -1, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zspsv", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Zspsv(Upper, 0, -1, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zspsv", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Zspsv(Upper, 2, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zspsv", err)

		//        Zspsvx
		*srnamt = "Zspsvx"
		*errt = fmt.Errorf("!nofact && fact != 'F': fact='/'")
		_, _, err = golapack.Zspsvx('/', Upper, 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, err = golapack.Zspsvx('N', '/', 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zspsvx('N', Upper, -1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, _, err = golapack.Zspsvx('N', Upper, 0, -1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, _, err = golapack.Zspsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, _, err = golapack.Zspsvx('N', Upper, 2, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, rw)
		chkxer2("Zspsvx", err)
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
