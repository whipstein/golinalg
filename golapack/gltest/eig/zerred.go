package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerred tests the error exits for the eigenvalue driver routines for
// DOUBLE COMPLEX PRECISION matrices:
//
// PATH  driver   description
// ----  ------   -----------
// ZEV   ZGEEV    find eigenvalues/eigenvectors for nonsymmetric A
// ZES   ZGEES    find eigenvalues/Schur form for nonsymmetric A
// ZVX   ZGEEVX   ZGEEV + balancing and condition estimation
// ZSX   ZGEESX   ZGEES + balancing and condition estimation
// ZBD   ZGESVD   compute SVD of an M-by-N matrix A
//       ZGESDD   compute SVD of an M-by-N matrix A(by divide and
//                conquer)
//       ZGEJSV   compute SVD of an M-by-N matrix A where M >= N
//       ZGESVDX  compute SVD of an M-by-N matrix A(by bisection
//                and inverse iteration)
//       ZGESVDQ  compute SVD of an M-by-N matrix A(with a
//                QR-Preconditioned )
func Zerred(path []byte, t *testing.T) {
	var abnrm, one, zero float64
	var i, ihi, ilo, info, j, lw, nmax, ns, nt, sdim int

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
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
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

	if string(c2) == "EV" {
		//        Test ZGEEV
		*srnamt = "ZGEEV "
		(*infot) = 1
		golapack.Zgeev('X', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgeev('N', 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgeev('N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgeev('N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgeev('V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgeev('N', 'V', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zgeev('V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEV ", &info, lerr, ok, t)
		nt = nt + 7

	} else if string(c2) == "ES" {
		//        Test ZGEES
		*srnamt = "ZGEES "
		(*infot) = 1
		golapack.Zgees('X', 'N', Zslect, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgees('N', 'X', Zslect, func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgees('N', 'S', Zslect, toPtr(-1), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zgees('N', 'S', Zslect, func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgees('V', 'S', Zslect, func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 4; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zgees('N', 'S', Zslect, func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEES ", &info, lerr, ok, t)
		nt = nt + 6

	} else if string(c2) == "VX" {
		//        Test ZGEEVX
		*srnamt = "ZGEEVX"
		(*infot) = 1
		golapack.Zgeevx('X', 'N', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgeevx('N', 'X', 'N', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgeevx('N', 'N', 'X', 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgeevx('N', 'N', 'N', 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgeevx('N', 'N', 'N', 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgeevx('N', 'N', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgeevx('N', 'V', 'N', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zgeevx('N', 'N', 'V', 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 4; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 20
		golapack.Zgeevx('N', 'N', 'N', 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		(*infot) = 20
		golapack.Zgeevx('N', 'N', 'V', 'V', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), x, vl, func() *int { y := 1; return &y }(), vr, func() *int { y := 1; return &y }(), &ilo, &ihi, s, &abnrm, r1, r2, w, func() *int { y := 2; return &y }(), rw, &info)
		Chkxer("ZGEEVX", &info, lerr, ok, t)
		nt = nt + 10

	} else if string(c2) == "SX" {
		//        Test ZGEESX
		*srnamt = "ZGEESX"
		(*infot) = 1
		golapack.Zgeesx('X', 'N', Zslect, 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgeesx('N', 'X', Zslect, 'N', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgeesx('N', 'N', Zslect, 'X', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgeesx('N', 'N', Zslect, 'N', toPtr(-1), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgeesx('N', 'N', Zslect, 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 4; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zgeesx('V', 'N', Zslect, 'N', func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 4; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		(*infot) = 15
		golapack.Zgeesx('N', 'N', Zslect, 'N', func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &sdim, x, vl, func() *int { y := 1; return &y }(), r1.GetPtr(0), r2.GetPtr(0), w, func() *int { y := 1; return &y }(), rw, &b, &info)
		Chkxer("ZGEESX", &info, lerr, ok, t)
		nt = nt + 7

	} else if string(c2) == "BD" {
		//        Test ZGESVD
		*srnamt = "ZGESVD"
		(*infot) = 1
		golapack.Zgesvd('X', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgesvd('N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgesvd('O', 'O', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgesvd('N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgesvd('N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zgesvd('N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zgesvd('A', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zgesvd('N', 'A', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &info)
		Chkxer("ZGESVD", &info, lerr, ok, t)
		nt = nt + 8
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test ZGESDD
		*srnamt = "ZGESDD"
		(*infot) = 1
		golapack.Zgesdd('X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgesdd('N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgesdd('N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgesdd('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgesdd('A', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgesdd('A', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 5; return &y }(), rw, &iw, &info)
		Chkxer("ZGESDD", &info, lerr, ok, t)
		nt = nt - 2
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test ZGEJSV
		*srnamt = "ZGEJSV"
		(*infot) = 1
		golapack.Zgejsv('X', 'U', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgejsv('G', 'X', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgejsv('G', 'U', 'X', 'R', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgejsv('G', 'U', 'V', 'X', 'N', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgejsv('G', 'U', 'V', 'R', 'X', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 13
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 2; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		(*infot) = 15
		golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), s, u, func() *int { y := 2; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &iw, &info)
		Chkxer("ZGEJSV", &info, lerr, ok, t)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test ZGESVDX
		*srnamt = "ZGESVDX"
		(*infot) = 1
		golapack.Zgesvdx('X', 'N', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgesvdx('N', 'X', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgesvdx('N', 'N', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgesvdx('N', 'N', 'A', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgesvdx('N', 'N', 'A', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgesvdx('N', 'N', 'A', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 8
		golapack.Zgesvdx('N', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), toPtrf64(-one), &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zgesvdx('N', 'N', 'V', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &one, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 10
		golapack.Zgesvdx('N', 'N', 'I', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 11
		golapack.Zgesvdx('V', 'N', 'I', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 15
		golapack.Zgesvdx('V', 'N', 'A', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		(*infot) = 17
		golapack.Zgesvdx('N', 'V', 'A', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &zero, &zero, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ns, s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, &iw, &info)
		Chkxer("ZGESVDX", &info, lerr, ok, t)
		nt = 12
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}

		//        Test ZGESVDQ
		*srnamt = "ZGESVDQ"
		(*infot) = 1
		golapack.Zgesvdq('X', 'P', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 2
		golapack.Zgesvdq('A', 'X', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 3
		golapack.Zgesvdq('A', 'P', 'X', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 4
		golapack.Zgesvdq('A', 'P', 'T', 'X', 'A', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 5
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'X', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 6
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 7
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 9
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), s, u, func() *int { y := 0; return &y }(), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 12
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, toPtr(-1), vt, func() *int { y := 0; return &y }(), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 14
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, toPtr(-1), &ns, &iw, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		(*infot) = 17
		golapack.Zgesvdq('A', 'P', 'T', 'A', 'A', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), s, u, func() *int { y := 1; return &y }(), vt, func() *int { y := 1; return &y }(), &ns, &iw, toPtr(-5), w, func() *int { y := 1; return &y }(), rw, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGESVDQ", &info, lerr, ok, t)
		nt = 11
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}
	}

	//     Print a summary line.
	if string(c2) != "BD" {
		if *ok {
			fmt.Printf(" %s passed the tests of the error exits (%3d tests done)\n", (*srnamt)[1:], nt)
		} else {
			fmt.Printf(" *** %s failed the tests of the error exits ***\n", (*srnamt)[1:])
		}
	}
}
