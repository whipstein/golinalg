package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derred tests the error exits for the eigenvalue driver routines for
// DOUBLE PRECISION matrices:
//
// PATH  driver   description
// ----  ------   -----------
// SEV   Dgeev   find eigenvalues/eigenvectors for nonsymmetric A
// SES   Dgees   find eigenvalues/Schur form for nonsymmetric A
// SVX   Dgeevx   SGEEV + balancing and condition estimation
// SSX   Dgeesx   SGEES + balancing and condition estimation
// DBD   Dgesvd   compute SVD of an M-by-N matrix A
//       Dgesdd   compute SVD of an M-by-N matrix A (by divide and
//                conquer)
//       Dgejsv   compute SVD of an M-by-N matrix A where M >= N
//       Dgesvdx  compute SVD of an M-by-N matrix A(by bisection
//                and inverse iteration)
//       Dgesvdq  compute SVD of an M-by-N matrix A(with a
//                QR-Preconditioned )
func derred(path string, t *testing.T) (nt int) {
	var one, zero float64
	var i, j, nmax int
	var err error

	nmax = 4
	one = 1.0
	zero = 0.0
	b := make([]bool, 4)
	r1 := vf(4)
	r2 := vf(4)
	s := vf(4)
	w := vf(nmax)
	wi := vf(4)
	wr := vf(4)
	iw := make([]int, 2*nmax)
	a := mf(4, 4, opts)
	u := mf(4, 4, opts)
	vl := mf(4, 4, opts)
	vr := mf(4, 4, opts)
	vt := mf(4, 4, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]

	//     Initialize A
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= nmax; i++ {
		a.Set(i-1, i-1, one)
	}
	(*ok) = true
	nt = 0

	if c2 == "ev" {
		//        Test DGEEV
		*srnamt = "Dgeev"
		*errt = fmt.Errorf("(!wantvl) && jobvl != 'N': jobvl='X'")
		_, err = golapack.Dgeev('X', 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("(!wantvr) && jobvr != 'N': jobvr='X'")
		_, err = golapack.Dgeev('N', 'X', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgeev('N', 'N', -1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dgeev('N', 'N', 2, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 6)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (wantvl && vl.Rows < n): jobvl='V', vl.Rows=1, n=2")
		_, err = golapack.Dgeev('V', 'N', 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 8)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (wantvr && vr.Rows < n): jobvr='V', vr.Rows=1, n=2")
		_, err = golapack.Dgeev('N', 'V', 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 8)
		chkxer2("Dgeev", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=3, minwrk=4, lquery=false")
		_, err = golapack.Dgeev('V', 'V', 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), w, 3)
		chkxer2("Dgeev", err)
		nt = nt + 7

	} else if c2 == "es" {
		//        Test DGEES
		*srnamt = "Dgees"
		*errt = fmt.Errorf("(!wantvs) && jobvs != 'N': jobvs='X'")
		_, _, err = golapack.Dgees('X', 'N', dslect, 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &b)
		chkxer2("Dgees", err)
		*errt = fmt.Errorf("(!wantst) && sort != 'N': sort='X'")
		_, _, err = golapack.Dgees('N', 'X', dslect, 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &b)
		chkxer2("Dgees", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgees('N', 'S', dslect, -1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &b)
		chkxer2("Dgees", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, err = golapack.Dgees('N', 'S', dslect, 2, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 6, &b)
		chkxer2("Dgees", err)
		*errt = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='V', vs.Rows=1, n=2")
		_, _, err = golapack.Dgees('V', 'S', dslect, 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 6, &b)
		chkxer2("Dgees", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=2, minwrk=3, lquery=false")
		_, _, err = golapack.Dgees('N', 'S', dslect, 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 2, &b)
		chkxer2("Dgees", err)
		nt = nt + 6

	} else if c2 == "vx" {
		//        Test Dgeevx
		*srnamt = "Dgeevx"
		*errt = fmt.Errorf("!(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B'): balanc='X'")
		_, _, _, _, err = golapack.Dgeevx('X', 'N', 'N', 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("(!wantvl) && (jobvl != 'N'): jobvl='X'")
		_, _, _, _, err = golapack.Dgeevx('N', 'X', 'N', 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("(!wantvr) && (jobvr != 'N'): jobvr='X'")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'X', 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("!(wntsnn || wntsne || wntsnb || wntsnv) || ((wntsne || wntsnb) && !(wantvl && wantvr)): sense='X'")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'N', 'X', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'N', 'N', -1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'N', 'N', 2, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("vl.Rows < 1 || (wantvl && vl.Rows < n): jobvl='V', vl.Rows=1, n=2")
		_, _, _, _, err = golapack.Dgeevx('N', 'V', 'N', 'N', 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 6, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("vr.Rows < 1 || (wantvr && vr.Rows < n): jobvr='V', vr.Rows=1, n=2")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'V', 'N', 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 6, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=1, minwrk=2, lquery=false")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'N', 'N', 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 1, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=2, minwrk=3, lquery=false")
		_, _, _, _, err = golapack.Dgeevx('N', 'V', 'N', 'N', 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 2, &iw)
		chkxer2("Dgeevx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=3, minwrk=7, lquery=false")
		_, _, _, _, err = golapack.Dgeevx('N', 'N', 'V', 'V', 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), vr.Off(0, 0).UpdateRows(1), s, r1, r2, w, 3, &iw)
		chkxer2("Dgeevx", err)
		nt = nt + 11

	} else if c2 == "sx" {
		//        Test Dgeesx
		*srnamt = "Dgeesx"
		*errt = fmt.Errorf("(!wantvs) && (jobvs != 'N'): jobvs='X'")
		_, _, _, _, err = golapack.Dgeesx('X', 'N', dslect, 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("(!wantst) && (sort != 'N'): sort='X'")
		_, _, _, _, err = golapack.Dgeesx('N', 'X', dslect, 'N', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("!(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn): sense='X'")
		_, _, _, _, err = golapack.Dgeesx('N', 'N', dslect, 'X', 0, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, _, err = golapack.Dgeesx('N', 'N', dslect, 'N', -1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 1, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, _, err = golapack.Dgeesx('N', 'N', dslect, 'N', 2, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 6, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='V', vs.Row=1, n=2")
		_, _, _, _, err = golapack.Dgeesx('V', 'N', dslect, 'N', 2, a.Off(0, 0).UpdateRows(2), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 6, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		*errt = fmt.Errorf("lwork < minwrk && !lquery: lwork=2, minwrk=3, lquery=false")
		_, _, _, _, err = golapack.Dgeesx('N', 'N', dslect, 'N', 1, a.Off(0, 0).UpdateRows(1), wr, wi, vl.Off(0, 0).UpdateRows(1), w, 2, &iw, 1, &b)
		chkxer2("Dgeesx", err)
		nt = nt + 7

	} else if c2 == "bd" {
		//        Test Dgesvd
		*srnamt = "Dgesvd"
		*errt = fmt.Errorf("!(wntua || wntus || wntuo || wntun): jobu='X'")
		_, err = golapack.Dgesvd('X', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo): jobvt='X'")
		_, err = golapack.Dgesvd('N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo): jobvt='O'")
		_, err = golapack.Dgesvd('O', 'O', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgesvd('N', 'N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgesvd('N', 'N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgesvd('N', 'N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("u.Rows < 1 || (wntuas && u.Rows < m): jobu='A', u.Rows=1, m=2")
		_, err = golapack.Dgesvd('A', 'N', 2, 1, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5)
		chkxer2("Dgesvd", err)
		*errt = fmt.Errorf("vt.Rows < 1 || (wntva && vt.Rows < n) || (wntvs && vt.Rows < minmn): jobvt='A', vt.Rows=1, m=1, n=2")
		_, err = golapack.Dgesvd('N', 'A', 1, 2, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5)
		chkxer2("Dgesvd", err)
		nt = 8
		// if *ok {
		// 	fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		// } else {
		// 	fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		// }

		//        Test Dgesdd
		*srnamt = "Dgesdd"
		*errt = fmt.Errorf("!(wntqa || wntqs || wntqo || wntqn): jobz='X'")
		_, err = golapack.Dgesdd('X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesdd", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgesdd('N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesdd", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dgesdd('N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesdd", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, err = golapack.Dgesdd('N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, &iw)
		chkxer2("Dgesdd", err)
		*errt = fmt.Errorf("u.Rows < 1 || (wntqas && u.Rows < m) || (wntqo && m < n && u.Rows < m): jobz='A', u.Rows=1, m=2, n=1")
		_, err = golapack.Dgesdd('A', 2, 1, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, &iw)
		chkxer2("Dgesdd", err)
		*errt = fmt.Errorf("vt.Rows < 1 || (wntqa && vt.Rows < n) || (wntqs && vt.Rows < minmn) || (wntqo && m >= n && vt.Rows < n): jobz='A', vt.Rows=1, n=2")
		_, err = golapack.Dgesdd('A', 1, 2, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 5, &iw)
		chkxer2("Dgesdd", err)
		nt = 6
		// if *ok {
		// 	fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		// } else {
		// 	fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		// }

		//        Test Dgejsv
		*srnamt = "Dgejsv"
		*errt = fmt.Errorf("!(rowpiv || l2rank || l2aber || errest || joba == 'C'): joba='X'")
		_, err = golapack.Dgejsv('X', 'U', 'V', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("!(lsvec || jobu == 'N' || jobu == 'W'): jobu='X'")
		_, err = golapack.Dgejsv('G', 'X', 'V', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("!(rsvec || jobv == 'N' || jobv == 'W') || (jracc && (!lsvec)): jobv='X'")
		_, err = golapack.Dgejsv('G', 'U', 'X', 'R', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("!(l2kill || defr): jobr='X'")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'X', 'N', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("!(l2tran || jobt == 'N'): jobt='X'")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'X', 'N', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("!(l2pert || jobp == 'N'): jobp='X'")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("(n < 0) || (n > m): m=0, n=-1")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', 0, -1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("a.Rows < m: a.Rows=1, m=2")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("lsvec && (u.Rows < m): jobu='U', u.Rows=1, m=2")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 2, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(2), w, 1, &iw)
		chkxer2("Dgejsv", err)
		*errt = fmt.Errorf("rsvec && (v.Rows < n): jobv='V', v.Rows=1, n=2")
		_, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', 2, 2, a.Off(0, 0).UpdateRows(2), s, u.Off(0, 0).UpdateRows(2), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgejsv", err)
		nt = 11
		// if *ok {
		// 	fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		// } else {
		// 	fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		// }

		//        Test Dgesvdx
		*srnamt = "Dgesvdx"
		*errt = fmt.Errorf("jobu != 'V' && jobu != 'N': jobu='X'")
		_, _, err = golapack.Dgesvdx('X', 'N', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("jobvt != 'V' && jobvt != 'N': jobvt='X'")
		_, _, err = golapack.Dgesvdx('N', 'X', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("!(alls || vals || inds): _range='X'")
		_, _, err = golapack.Dgesvdx('N', 'N', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Dgesvdx('N', 'N', 'A', -1, 0, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, err = golapack.Dgesvdx('N', 'N', 'A', 0, -1, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("m > a.Rows: a.Rows=1, m=2")
		_, _, err = golapack.Dgesvdx('N', 'N', 'A', 2, 1, a.Off(0, 0).UpdateRows(1), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("vl < zero: _range='V', vl=-1")
		_, _, err = golapack.Dgesvdx('N', 'N', 'V', 2, 1, a.Off(0, 0).UpdateRows(2), -one, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("vu <= vl: _range='V', vl=1, vu=0")
		_, _, err = golapack.Dgesvdx('N', 'N', 'V', 2, 1, a.Off(0, 0).UpdateRows(2), one, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("il < 1 || il > max(1, minmn): _range='I', il=0, m=2, n=2")
		_, _, err = golapack.Dgesvdx('N', 'N', 'I', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 1, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("iu < min(minmn, il) || iu > minmn: _range='I', il=1, iu=0, m=2, n=2")
		_, _, err = golapack.Dgesvdx('V', 'N', 'I', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 1, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		*errt = fmt.Errorf("wantu && u.Rows < m: jobu='V', u.Rows=1, m=2")
		_, _, err = golapack.Dgesvdx('V', 'N', 'A', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		// *errt = fmt.Errorf("vt.Rows < iu-il+1: jobvt='V', _range='A', vt.Rows=1, il=0, iu=0")
		*errt = fmt.Errorf("vt.Rows < minmn: jobvt='V', _range='A', vt.Rows=1, m=2, n=2")
		_, _, err = golapack.Dgesvdx('N', 'V', 'A', 2, 2, a.Off(0, 0).UpdateRows(2), zero, zero, 0, 0, s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), w, 1, &iw)
		chkxer2("Dgesvdx", err)
		nt = 12
		// if *ok {
		// 	fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		// } else {
		// 	fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		// }

		//        Test Dgesvdq
		*srnamt = "Dgesvdq"
		*errt = fmt.Errorf("!(accla || acclm || acclh): joba='X'")
		_, _, err = golapack.Dgesvdq('X', 'P', 'T', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("!(rowprm || jobp == 'N'): jobp='X'")
		_, _, err = golapack.Dgesvdq('A', 'X', 'T', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("!(rtrans || jobr == 'N'): jobr='X'")
		_, _, err = golapack.Dgesvdq('A', 'P', 'X', 'A', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("!(lsvec || dntwu): jobu='X'")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'X', 'A', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		// *errt = fmt.Errorf("wntur && wntva: jobv='X'")
		*errt = fmt.Errorf("!(rsvec || dntwv): jobv='X'")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'X', 0, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("m < 0: m=-1")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', -1, 0, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("(n < 0) || (n > m): m=0, n=1")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', 0, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', 2, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("u.Rows < 1 || (lsvc0 && u.Rows < m) || (wntuf && u.Rows < n): jobu='A', u.Rows=-1, m=1, n=1")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(-1), vt.Off(0, 0).UpdateRows(1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("v.Rows < 1 || (rsvec && v.Rows < n) || (conda && v.Rows < n): jobv='A', v.Rows=-1, n=1")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(-1), &iw, 1, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		*errt = fmt.Errorf("liwork < iminwrk && !lquery: liwork=-5, iminwrk=1, lquery=false")
		_, _, err = golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', 1, 1, a.Off(0, 0).UpdateRows(1), s, u.Off(0, 0).UpdateRows(1), vt.Off(0, 0).UpdateRows(1), &iw, -5, w, 1, w, 1)
		chkxer2("Dgesvdq", err)
		nt = 11
		// if *ok {
		// 	fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		// } else {
		// 	fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		// }
	}

	//     Print a summary line.
	// if c2 != "bd" {
	// 	if *ok {
	// 		fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
	// 	} else {
	// 		fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
	// 	}
	// }

	if !(*ok) {
		t.Fail()
	}

	return
}
