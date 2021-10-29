package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerred tests the error exits for the eigenvalue driver routines for
// DOUBLE COMPLEX PRECISION matrices:
//
// PATH  driver   description
// ----  ------   -----------
// ZEV   Zgeev   find eigenvalues/eigenvectors for nonsymmetric A
// ZES   Zgees   find eigenvalues/Schur form for nonsymmetric A
// ZVX   Zgeevx   Zgeev+ balancing and condition estimation
// ZSX   Zgeesx   Zgees+ balancing and condition estimation
// ZBD   Zgesvd   compute SVD of an M-by-N matrix A
//       Zgesdd   compute SVD of an M-by-N matrix A(by divide and
//                conquer)
//       Zgejsv   compute SVD of an M-by-N matrix A where M >= N
//       Zgesvdx  compute SVD of an M-by-N matrix A(by bisection
//                and inverse iteration)
//       Zgesvdq  compute SVD of an M-by-N matrix A(with a
//                QR-Preconditioned )
func zerred(path string, t *testing.T) {
	var one, zero float64
	var i, j, lw, nmax, ns, nt int
	var err error

	nmax = 4
	lw = 5 * nmax
	one = 1.0
	zero = 0.0
	b := make([]bool, 4)
	w := cvf(lw)
	x := cvf(4)
	r1 := vf(4)
	r2 := vf(4)
	rw := vf(lw)
	s := vf(4)
	iw := make([]int, lw)
	a := cmf(4, 4, opts)
	u := cmf(4, 4, opts)
	vl := cmf(4, 4, opts)
	vr := cmf(4, 4, opts)
	vt := cmf(4, 4, opts)

	errt := &gltest.Common.Infoc.Errt
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Initialize A
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.SetRe(i-1, i-1, one)
	}
	*ok = true
	nt = 0

	if c2 == "ev" {
		//        Test ZGEEV
		*srnamt = "Zgeev"
		*errt = fmt.Errorf("(!wantvl) && (jobvl != 'N'): jobvl='X'")
		_, err = golapack.Zgeev('X', 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("(!wantvr) && (jobvr != 'N'): jobvr='X'")
		_, err = golapack.Zgeev('N', 'X', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgeev('N', 'N', -1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zgeev('N', 'N', 2, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 4, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (wantvl && vl.Rows < n): jobvl='V', vl.Rows=1, n=2")
		_, err = golapack.Zgeev('V', 'N', 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 4, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (wantvr && vr.Rows < n): jobvr='V', vr.Rows=1, n=2")
		_, err = golapack.Zgeev('N', 'V', 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 4, rw)
		chkxer2("Zgeev", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=2, lquery=false")
		_, err = golapack.Zgeev('V', 'V', 1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgeev", err)
		nt = nt + 7

	} else if c2 == "es" {
		//        Test ZGEES
		*srnamt = "Zgees"
		*errt = fmt.Errorf("(!wantvs) && (jobvs != 'N'): jobvs='X'")
		_, _, err = golapack.Zgees('X', 'N', zslect, 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgees", err)
		*errt = fmt.Errorf("(!wantst) && (sort != 'N'): sort='X'")
		_, _, err = golapack.Zgees('N', 'X', zslect, 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgees", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Zgees('N', 'S', zslect, -1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgees", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Zgees('N', 'S', zslect, 2, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 4, rw, &b)
		chkxer2("Zgees", err)
		*errt = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='V', vs.Rows=1, n=2")
		_, _, err = golapack.Zgees('V', 'S', zslect, 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), w, 4, rw, &b)
		chkxer2("Zgees", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=2, lquery=false")
		_, _, err = golapack.Zgees('N', 'S', zslect, 1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgees", err)
		nt = nt + 6

	} else if c2 == "vx" {
		//        Test Zgeevx
		*srnamt = "Zgeevx"
		*errt = fmt.Errorf("!(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B'): balanc='X'")
		_, _, _, _, err = golapack.Zgeevx('X', 'N', 'N', 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("(!wantvl) && (jobvl != 'N'): jobvl='X'")
		_, _, _, _, err = golapack.Zgeevx('N', 'X', 'N', 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("(!wantvr) && (jobvr != 'N'): jobvr='X'")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'X', 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("!(wntsnn || wntsne || wntsnb || wntsnv) || ((wntsne || wntsnb) && !(wantvl && wantvr)): sense='X'")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'N', 'X', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'N', 'N', -1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'N', 'N', 2, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 4, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (wantvl && vl.Rows < n): jobvl='V', vl.Rows=1, n=2")
		_, _, _, _, err = golapack.Zgeevx('N', 'V', 'N', 'N', 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 4, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (wantvr && vr.Rows < n): jobvr='V', vr.Rows=1, n=2")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'V', 'N', 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 4, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=2, lquery=false")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'N', 'N', 1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, rw)
		chkxer2("Zgeevx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=2, minwrk=3, lquery=false")
		_, _, _, _, err = golapack.Zgeevx('N', 'N', 'V', 'V', 1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 2, rw)
		chkxer2("Zgeevx", err)
		nt = nt + 10

	} else if c2 == "sx" {
		//        Test Zgeesx
		*srnamt = "Zgeesx"
		*errt = fmt.Errorf("(!wantvs) && (jobvs != 'N'): jobvs='X'")
		_, _, _, _, err = golapack.Zgeesx('X', 'N', zslect, 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("(!wantst) && (sort != 'N'): sort='X'")
		_, _, _, _, err = golapack.Zgeesx('N', 'X', zslect, 'N', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("!(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn): sense='X'")
		_, _, _, _, err = golapack.Zgeesx('N', 'N', zslect, 'X', 0, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Zgeesx('N', 'N', zslect, 'N', -1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, _, err = golapack.Zgeesx('N', 'N', zslect, 'N', 2, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 4, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='V', vs.Rows=1, n=2")
		_, _, _, _, err = golapack.Zgeesx('V', 'N', zslect, 'N', 2, a.Off(0, 0).UpdateRows(2), x, vl.Off(0, 0).UpdateRows(1), w, 4, rw, &b)
		chkxer2("Zgeesx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=2, lquery=false")
		_, _, _, _, err = golapack.Zgeesx('N', 'N', zslect, 'N', 1, a.Off(0, 0).UpdateRows(1), x, vl.Off(0, 0).UpdateRows(1), w, 1, rw, &b)
		chkxer2("Zgeesx", err)
		nt = nt + 7

	} else if c2 == "bd" {
		//        Test Zgesvd
		*srnamt = "Zgesvd"
		*errt = fmt.Errorf("!(wntua || wntus || wntuo || wntun): jobu='X'")
		_, err = golapack.Zgesvd('X', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo): jobvt='X'")
		_, err = golapack.Zgesvd('N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo): jobvt='O'")
		_, err = golapack.Zgesvd('O', 'O', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgesvd('N', 'N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgesvd('N', 'N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgesvd('N', 'N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("u.Rows < 1 || (wntuas && u.Rows < m): jobu='A', u.Rows=1, m=2")
		_, err = golapack.Zgesvd('A', 'N', 2, 1, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw)
		chkxer2("Zgesvd", err)
		*errt = fmt.Errorf("vt.Rows < 1 || (wntva && vt.Rows < n) || (wntvs && vt.Rows < minmn): jobvt='A', vt.Rows=1, n=2")
		_, err = golapack.Zgesvd('N', 'A', 1, 2, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw)
		chkxer2("Zgesvd", err)
		nt = nt + 8
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test Zgesdd
		*srnamt = "Zgesdd"
		*errt = fmt.Errorf("!(wntqa || wntqs || wntqo || wntqn): jobz='X'")
		_, err = golapack.Zgesdd('X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesdd", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgesdd('N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesdd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgesdd('N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesdd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Zgesdd('N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw, &iw)
		chkxer2("Zgesdd", err)
		*errt = fmt.Errorf("u.Rows < 1 || (wntqas && u.Rows < m) || (wntqo && m < n && u.Rows < m): jobz='A', u.Rows=1, m=2, n=1")
		_, err = golapack.Zgesdd('A', 2, 1, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw, &iw)
		chkxer2("Zgesdd", err)
		*errt = fmt.Errorf("vt.Rows < 1 || (wntqa && vt.Rows < n) || (wntqs && vt.Rows < minmn) || (wntqo && m >= n && vt.Rows < n): jobz='A', vt.Rows=1, m=1, n=2")
		_, err = golapack.Zgesdd('A', 1, 2, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, rw, &iw)
		chkxer2("Zgesdd", err)
		nt = nt - 2
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test Zgejsv
		*srnamt = "Zgejsv"
		// *errt = fmt.Errorf("lwork < minwrk && (!lquery): lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		// *errt = fmt.Errorf("lrwork < minrwrk && (!lquery): lrwork=%v, minrwrk=%v, lquery=%v", lrwork, minrwrk, lquery)
		*errt = fmt.Errorf("!(rowpiv || l2rank || l2aber || errest || joba == 'C'): joba='X'")
		_, err = golapack.Zgejsv('X', 'U', 'V', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("!(lsvec || jobu == 'N' || (jobu == 'W' && rsvec && l2tran)): jobu='X', jobv='V', jobt='N'")
		_, err = golapack.Zgejsv('G', 'X', 'V', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("!(rsvec || jobv == 'N' || (jobv == 'W' && lsvec && l2tran)): jobu='U', jobv='X', jobt='N'")
		_, err = golapack.Zgejsv('G', 'U', 'X', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("!(l2kill || defr): jobr='X'")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'X', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("!(jobt == 'T' || jobt == 'N'): jobt='X'")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'X', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("!(l2pert || jobp == 'N'): jobp='X'")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("(n < 0) || (n > m): m=0, n=-1")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("a.Rows < m: a.Rows=1, m=2")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("lsvec && (u.Rows < m): jobu='U', u.Rows=1, m=2")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 2, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(2), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		*errt = fmt.Errorf("rsvec && (v.Rows < n): jobv='V', v.Rows=1, n=2")
		_, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 2, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(2), vt.Off(0, 0).UpdateRows(1), w, 1, rw, 1, &iw)
		chkxer2("Zgejsv", err)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test Zgesvdx
		*srnamt = "Zgesvdx"
		*errt = fmt.Errorf("jobu != 'V' && jobu != 'N': jobu='X'")
		_, err = golapack.Zgesvdx('X', 'N', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("jobvt != 'V' && jobvt != 'N': jobvt='X'")
		_, err = golapack.Zgesvdx('N', 'X', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("!(alls || vals || inds): _range='X'")
		_, err = golapack.Zgesvdx('N', 'N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Zgesvdx('N', 'N', 'A', -1, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zgesvdx('N', 'N', 'A', 0, -1, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("m > a.Rows: a.Rows=1, m=2")
		_, err = golapack.Zgesvdx('N', 'N', 'A', 2, 1, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("vl < zero: vl=-1")
		_, err = golapack.Zgesvdx('N', 'N', 'V', 2, 1, a.Off(0, 0).UpdateRows(2), -one, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("vu <= vl: vl=1, vu=0")
		_, err = golapack.Zgesvdx('N', 'N', 'V', 2, 1, a.Off(0, 0).UpdateRows(2), one, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("il < 1 || il > max(1, minmn): il=0, minmn=2")
		_, err = golapack.Zgesvdx('N', 'N', 'I', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 1, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("iu < min(minmn, il) || iu > minmn: il=1, iu=0, minmn=2")
		_, err = golapack.Zgesvdx('V', 'N', 'I', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 1, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		*errt = fmt.Errorf("wantu && u.Rows < m: jobu='V', u.Rows=1, m=2")
		_, err = golapack.Zgesvdx('V', 'N', 'A', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		// *errt = fmt.Errorf("vt.Rows < iu-il+1: vt.Rows=1, il=0, iu=0")
		*errt = fmt.Errorf("vt.Rows < minmn: vt.Rows=1, minmn=2")
		_, err = golapack.Zgesvdx('N', 'V', 'A', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 0, ns, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, rw, &iw)
		chkxer2("Zgesvdx", err)
		nt = 12
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test Zgesvdq
		*srnamt = "Zgesvdq"
		*errt = fmt.Errorf("!(accla || acclm || acclh): joba='X'")
		_, _, _, err = golapack.Zgesvdq('X', 'P', 'T', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("!(rowprm || jobp == 'N'): jobp='X'")
		_, _, _, err = golapack.Zgesvdq('A', 'X', 'T', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("!(rtrans || jobr == 'N'): jobr='X'")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'X', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("!(lsvec || dntwu): jobu='X'")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'X', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		// *errt = fmt.Errorf("wntur && wntva: jobu='A', jobv='X'")
		*errt = fmt.Errorf("!(rsvec || dntwv): jobv='X'")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		// *errt = fmt.Errorf("(n < 0) || (n > m): m=0, n=1")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("(n < 0) || (n > m): m=0, n=1")
		// *errt = fmt.Errorf("lcwork < minwrk && (!lquery): lcwork=1, minwrk=4, lquery=false")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', 0, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		// *errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=0, m=1")
		*errt = fmt.Errorf("lcwork < minwrk && (!lquery): lcwork=1, minwrk=4, lquery=false")
		// *errt = fmt.Errorf("m < 0: m=-1")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("u.Rows < 1 || (lsvc0 && u.Rows < m) || (wntuf && u.Rows < n): jobu='A', u.Rows=-1, m=1, n=1")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(-1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("v.Rows < 1 || (rsvec && v.Rows < n) || (conda && v.Rows < n): jobv='A', v.Rows=-1, n=1")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(-1), &iw, 1, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		*errt = fmt.Errorf("liwork < iminwrk && !lquery: liwork=-5, iminwrk=1, lquery=false")
		_, _, _, err = golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, -5, w, 1, rw, 1)
		chkxer2("Zgesvdq", err)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}
	}

	//     Print a summary line.
	if c2 != "bd" {
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}
	}
	*infot = 0
	*srnamt = ""
	if !(*ok) {
		t.Fail()
	}
}
