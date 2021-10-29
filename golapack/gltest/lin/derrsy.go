package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrsy tests the error exits for the DOUBLE PRECISION routines
// for symmetric indefinite matrices.
func derrsy(path string, t *testing.T) {
	var anrm float64
	var i, j, nmax int
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4

	a := mf(4, 4, opts)
	af := mf(4, 4, opts)
	ap := vf(4 * 4)
	afp := vf(4 * 4)
	b := mf(4, 1, opts)
	e := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	w := vf(3 * nmax)
	x := mf(4, 1, opts)
	ip := make([]int, 4)
	iw := make([]int, 4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		e.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.SetIdx(j-1, 0.)
		ip[j-1] = j
		iw[j-1] = j
	}
	anrm = 1.0
	*ok = true

	if c2 == "sy" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//        Dsytrf
		*srnamt = "Dsytrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytrf('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Dsytrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytrf(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Dsytrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytrf(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("Dsytrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Dsytrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("Dsytrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Dsytrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("Dsytrf", err)

		//        Dsytf2
		*srnamt = "Dsytf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytf2('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytf2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytf2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2", err)

		//        Dsytri
		*srnamt = "Dsytri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytri('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Dsytri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytri(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Dsytri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytri(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Dsytri", err)

		//        Dsytri2
		*srnamt = "Dsytri2"
		// *errt = fmt.Errorf("lwork < minsize && !lquery: lwork=%v, minsize=%v, lquery=%v", lwork, minsize, lquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytri2('/', 0, a.Off(0, 0).UpdateRows(1), &ip, x, iw[0])
		chkxer2("Dsytri2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytri2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, x, iw[0])
		chkxer2("Dsytri2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytri2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, x, iw[0])
		chkxer2("Dsytri2", err)

		//        Dsytri2x
		*srnamt = "Dsytri2x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytri2x('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Dsytri2x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytri2x(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Dsytri2x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytri2x(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Dsytri2x", err)

		//        Dsytrs
		*srnamt = "Dsytrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dsytrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dsytrs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dsytrs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dsytrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("Dsytrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dsytrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs", err)

		//        Dsyrfs
		*srnamt = "Dsyrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsyrfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsyrfs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dsyrfs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		_, err = golapack.Dsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		_, err = golapack.Dsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsyrfs", err)

		//        Dsycon
		*srnamt = "Dsycon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsycon('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("Dsycon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsycon(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("Dsycon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsycon(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("Dsycon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Dsycon(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -1.0, w, &iw)
		chkxer2("Dsycon", err)

	} else if c2 == "sr" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting.
		//
		//        DsytrfRook
		*srnamt = "DsytrfRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrfRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, x, 1)
		chkxer2("DsytrfRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrfRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, x, 1)
		chkxer2("DsytrfRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrfRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, x, 4)
		chkxer2("DsytrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.DsytrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, x, 0)
		chkxer2("DsytrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.DsytrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, x, -2)
		chkxer2("DsytrfRook", err)

		//        Dsytf2Rook
		*srnamt = "Dsytf2Rook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytf2Rook('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2Rook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytf2Rook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2Rook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytf2Rook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Dsytf2Rook", err)

		//        DsytriRook
		*srnamt = "DsytriRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytriRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("DsytriRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytriRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("DsytriRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytriRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("DsytriRook", err)

		//        DsytrsRook
		*srnamt = "DsytrsRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.DsytrsRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.DsytrsRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.DsytrsRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.DsytrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("DsytrsRook", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.DsytrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsRook", err)

		//        DsyconRook
		*srnamt = "DsyconRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsyconRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("DsyconRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsyconRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("DsyconRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsyconRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w, &iw)
		chkxer2("DsyconRook", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.DsyconRook(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -1.0, w, &iw)
		chkxer2("DsyconRook", err)

	} else if c2 == "sk" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//        DsytrfRk
		*srnamt = "DsytrfRk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrfRk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("DsytrfRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrfRk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("DsytrfRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrfRk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("DsytrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.DsytrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("DsytrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.DsytrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("DsytrfRk", err)

		//        Dsytf2Rk
		*srnamt = "Dsytf2Rk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytf2Rk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Dsytf2Rk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytf2Rk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Dsytf2Rk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytf2Rk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Dsytf2Rk", err)

		//        Dsytri3
		*srnamt = "Dsytri3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytri3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytri3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytri3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=0, lwkopt=8, lquery=false")
		_, err = golapack.Dsytri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("Dsytri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=-2, lwkopt=8, lquery=false")
		_, err = golapack.Dsytri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("Dsytri3", err)

		//        Dsytri3x
		*srnamt = "Dsytri3x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytri3x('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytri3x(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytri3x(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Dsytri3x", err)

		//        Dsytrs3
		*srnamt = "Dsytrs3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsytrs3('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsytrs3(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs3", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.Dsytrs3(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsytrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), e, &ip, b.Off(0, 0).UpdateRows(2))
		chkxer2("Dsytrs3", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.Dsytrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), e, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsytrs3", err)

		//        Dsycon3
		*srnamt = "Dsycon3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsycon3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w, &iw)
		chkxer2("Dsycon3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsycon3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w, &iw)
		chkxer2("Dsycon3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dsycon3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w, &iw)
		chkxer2("Dsycon3", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Dsycon3(Upper, 1, a.Off(0, 0).UpdateRows(1), e, &ip, -1.0, w, &iw)
		chkxer2("Dsycon3", err)

	} else if c2 == "sa" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        DsytrfAa
		*srnamt = "DsytrfAa"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrfAa('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("DsytrfAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrfAa(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("DsytrfAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrfAa(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("DsytrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=0, n=0, lquery=false")
		_, err = golapack.DsytrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("DsytrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=-2, n=0, lquery=false")
		_, err = golapack.DsytrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("DsytrfAa", err)

		//        DsytrsAa
		*srnamt = "DsytrsAa"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrsAa('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrsAa(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsytrsAa(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.Off(0, 0).UpdateRows(2), w, 1)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.DsytrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, 1)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=0, n=0, lquery=false")
		_, err = golapack.DsytrsAa(Upper, 0, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, 0)
		chkxer2("DsytrsAa", err)
		*errt = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=-2, n=0, lquery=false")
		_, err = golapack.DsytrsAa(Upper, 0, 1, a.Off(0, 0).UpdateRows(2), &ip, b.Off(0, 0).UpdateRows(1), w, -2)
		chkxer2("DsytrsAa", err)

	} else if c2 == "s2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        DsytrfAa2stage
		*srnamt = "DsytrfAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrfAa2stage('/', 0, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, w, 1)
		chkxer2("DsytrfAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrfAa2stage(Upper, -1, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, w, 1)
		chkxer2("DsytrfAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(1), w, 2, &ip, &ip, w, 1)
		chkxer2("DsytrfAa2stage", err)
		*errt = fmt.Errorf("ltb < 4*n && !tquery: ltb=1, n=2, tquery=false")
		_, err = golapack.DsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), w, 1, &ip, &ip, w, 1)
		chkxer2("DsytrfAa2stage", err)
		*errt = fmt.Errorf("lwork < n && !wquery: lwork=0, n=2, wquery=false")
		_, err = golapack.DsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), w, 8, &ip, &ip, w, 0)
		chkxer2("DsytrfAa2stage", err)

		//        DsytrsAa2stage
		*srnamt = "DsytrsAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.DsytrsAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.DsytrsAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.DsytrsAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.DsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), w, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)
		*errt = fmt.Errorf("ltb < (4 * n): ltb=1, n=2")
		_, err = golapack.DsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), w, 1, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.DsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), w, 8, &ip, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("DsytrsAa2stage", err)

	} else if c2 == "sp" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite packed matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//        Dsptrf
		*srnamt = "Dsptrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsptrf('/', 0, ap, &ip)
		chkxer2("Dsptrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsptrf(Upper, -1, ap, &ip)
		chkxer2("Dsptrf", err)

		//        Dsptri
		*srnamt = "Dsptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dsptri('/', 0, ap, &ip, w)
		chkxer2("Dsptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dsptri(Upper, -1, ap, &ip, w)
		chkxer2("Dsptri", err)

		//        Dsptrs
		*srnamt = "Dsptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dsptrs('/', 0, 0, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dsptrs(Upper, -1, 0, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dsptrs(Upper, 0, -1, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dsptrs(Upper, 2, 1, ap, &ip, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dsptrs", err)

		//        Dsprfs
		*srnamt = "Dsprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dsprfs('/', 0, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dsprfs(Upper, -1, 0, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dsprfs(Upper, 0, -1, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dsprfs(Upper, 2, 1, ap, afp, &ip, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dsprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dsprfs(Upper, 2, 1, ap, afp, &ip, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dsprfs", err)

		//        Dspcon
		*srnamt = "Dspcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dspcon('/', 0, ap, &ip, anrm, w, &iw)
		chkxer2("Dspcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dspcon(Upper, -1, ap, &ip, anrm, w, &iw)
		chkxer2("Dspcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Dspcon(Upper, 1, ap, &ip, -1.0, w, &iw)
		chkxer2("Dspcon", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
