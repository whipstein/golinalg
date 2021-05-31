package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derred tests the error exits for the eigenvalue driver routines for
// DOUBLE PRECISION matrices:
//
// PATH  driver   description
// ----  ------   -----------
// SEV   DGEEV    find eigenvalues/eigenvectors for nonsymmetric A
// SES   DGEES    find eigenvalues/Schur form for nonsymmetric A
// SVX   DGEEVX   SGEEV + balancing and condition estimation
// SSX   DGEESX   SGEES + balancing and condition estimation
// DBD   DGESVD   compute SVD of an M-by-N matrix A
//       DGESDD   compute SVD of an M-by-N matrix A (by divide and
//                conquer)
//       DGEJSV   compute SVD of an M-by-N matrix A where M >= N
//       DGESVDX  compute SVD of an M-by-N matrix A(by bisection
//                and inverse iteration)
//       DGESVDQ  compute SVD of an M-by-N matrix A(with a
//                QR-Preconditioned )
func Derred(path []byte, t *testing.T) {
	var abnrm, one, zero float64
	var i, ihi, ilo, info, j, nmax, ns, nt, sdim int

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

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
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

	if string(c2) == "EV" {
		//        Test DGEEV
		*srnamt = "DGEEV "
		*infot = 1
		golapack.Dgeev('X', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgeev('N', 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgeev('N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgeev('N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 6; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgeev('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 8; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgeev('N', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 8; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgeev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 3; return &y }(), &info)
		Chkxer("DGEEV ", &info, lerr, ok, t)
		nt = nt + 7

	} else if string(c2) == "ES" {
		//        Test DGEES
		*srnamt = "DGEES "
		*infot = 1
		golapack.Dgees('X', 'N', Dslect, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgees('N', 'X', Dslect, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgees('N', 'S', Dslect, toPtr(-1), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgees('N', 'S', Dslect, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 6; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgees('V', 'S', Dslect, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 6; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgees('N', 'S', Dslect, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 2; return &y }(), &b, &info)
		Chkxer("DGEES ", &info, lerr, ok, t)
		nt = nt + 6

	} else if string(c2) == "VX" {
		//        Test DGEEVX
		*srnamt = "DGEEVX"
		*infot = 1
		golapack.Dgeevx('X', 'N', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgeevx('N', 'X', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgeevx('N', 'N', 'X', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgeevx('N', 'N', 'N', 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgeevx('N', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgeevx('N', 'N', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgeevx('N', 'V', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 6; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgeevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 6; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 21
		golapack.Dgeevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 21
		golapack.Dgeevx('N', 'V', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 2; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		*infot = 21
		golapack.Dgeevx('N', 'N', 'V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), wr, wi, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 3; return &y }(), &iw, &info)
		Chkxer("DGEEVX", &info, lerr, ok, t)
		nt = nt + 11

	} else if string(c2) == "SX" {
		//        Test DGEESX
		*srnamt = "DGEESX"
		*infot = 1
		golapack.Dgeesx('X', 'N', Dslect, 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgeesx('N', 'X', Dslect, 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgeesx('N', 'N', Dslect, 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgeesx('N', 'N', Dslect, 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgeesx('N', 'N', Dslect, 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 6; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgeesx('V', 'N', Dslect, 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 6; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		*infot = 16
		golapack.Dgeesx('N', 'N', Dslect, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, wr, wi, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 2; return &y }(), &iw, func() *int { y := 1; return &y }(), &b, &info)
		Chkxer("DGEESX", &info, lerr, ok, t)
		nt = nt + 7

	} else if string(c2) == "BD" {
		//        Test DGESVD
		*srnamt = "DGESVD"
		*infot = 1
		golapack.Dgesvd('X', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesvd('N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesvd('O', 'O', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgesvd('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgesvd('N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgesvd('A', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgesvd('N', 'A', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &info)
		Chkxer("DGESVD", &info, lerr, ok, t)
		nt = 8
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}

		//        Test DGESDD
		*srnamt = "DGESDD"
		*infot = 1
		golapack.Dgesdd('X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesdd('N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgesdd('N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgesdd('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgesdd('A', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgesdd('A', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), &iw, &info)
		Chkxer("DGESDD", &info, lerr, ok, t)
		nt = 6
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}

		//        Test DGEJSV
		*srnamt = "DGEJSV"
		*infot = 1
		golapack.Dgejsv('X', 'U', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgejsv('G', 'X', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgejsv('G', 'U', 'X', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgejsv('G', 'U', 'V', 'X', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgejsv('G', 'U', 'V', 'R', 'X', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 13
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 2; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGEJSV", &info, lerr, ok, t)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}

		//        Test DGESVDX
		*srnamt = "DGESVDX"
		*infot = 1
		golapack.Dgesvdx('X', 'N', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesvdx('N', 'X', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgesvdx('N', 'N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgesvdx('N', 'N', 'A', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgesvdx('N', 'N', 'A', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgesvdx('N', 'N', 'A', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 8
		golapack.Dgesvdx('N', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), toPtrf64(-one), &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgesvdx('N', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &one, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 10
		golapack.Dgesvdx('N', 'N', 'I', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 11
		golapack.Dgesvdx('V', 'N', 'I', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 15
		golapack.Dgesvdx('V', 'N', 'A', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgesvdx('N', 'V', 'A', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("DGESVDX", &info, lerr, ok, t)
		nt = 12
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}

		//        Test DGESVDQ
		*srnamt = "DGESVDQ"
		*infot = 1
		golapack.Dgesvdq('X', 'P', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 2
		golapack.Dgesvdq('A', 'X', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 3
		golapack.Dgesvdq('A', 'P', 'X', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 4
		golapack.Dgesvdq('A', 'P', 'T', 'X', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 5
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 6
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 7
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 9
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 12
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, toPtr(-1), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 14
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, toPtr(-1), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		*infot = 17
		golapack.Dgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), &ns, &iw, toPtr(-5), w, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
		Chkxer("DGESVDQ", &info, lerr, ok, t)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}
	}

	//     Print a summary line.
	if string(c2) != "BD" {
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", *srnamt, nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", *srnamt)
		}
	}
}
