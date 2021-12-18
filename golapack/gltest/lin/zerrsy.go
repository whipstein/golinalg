package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

//zerrsy tests the error exits for the COMPLEX*16 routines
// for symmetric indefinite matrices.
func zerrsy(path string, t *testing.T) {
	var anrm float64
	var i, j, nmax int
	var err error

	nmax = 4
	b := cvf(4)
	e := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
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
		ip[j-1] = j
	}
	anrm = 1.0
	(*ok) = true

	if c2 == "sy" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//       Zsytrf
		*srnamt = "Zsytrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytrf('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zsytrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytrf(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zsytrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytrf(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("Zsytrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Zsytrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("Zsytrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Zsytrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("Zsytrf", err)

		//       Zsytf2
		*srnamt = "Zsytf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytf2('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytf2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytf2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2", err)

		//       Zsytri
		*srnamt = "Zsytri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytri('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zsytri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytri(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zsytri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytri(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zsytri", err)

		//       Zsytri2
		*srnamt = "Zsytri2"
		// *errt = fmt.Errorf("lwork < minsize && !lquery: lwork=%v, minsize=%v, lquery=%v", lwork, minsize, lquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytri2('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zsytri2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytri2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zsytri2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytri2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zsytri2", err)

		//       Zsytri2x
		*srnamt = "Zsytri2x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytri2x('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri2x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytri2x(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri2x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytri2x(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri2x", err)

		//       Zsytrs
		*srnamt = "Zsytrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zsytrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zsytrs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zsytrs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zsytrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("Zsytrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zsytrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs", err)

		//       Zsyrfs
		*srnamt = "Zsyrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zsyrfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zsyrfs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zsyrfs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Zsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zsyrfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsyrfs", err)

		//       Zsycon
		*srnamt = "Zsycon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsycon('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zsycon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsycon(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zsycon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsycon(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zsycon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zsycon(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -anrm, w)
		chkxer2("Zsycon", err)

	} else if c2 == "sr" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) diagonal pivoting method.
		//
		//       ZsytrfRook
		*srnamt = "ZsytrfRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsytrfRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZsytrfRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsytrfRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZsytrfRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsytrfRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("ZsytrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZsytrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("ZsytrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZsytrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("ZsytrfRook", err)

		//       Zsytf2Rook
		*srnamt = "Zsytf2Rook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytf2Rook('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2Rook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytf2Rook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2Rook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytf2Rook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zsytf2Rook", err)

		//       ZsytriRook
		*srnamt = "ZsytriRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsytriRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZsytriRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsytriRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZsytriRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsytriRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZsytriRook", err)

		//       ZsytrsRook
		*srnamt = "ZsytrsRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZsytrsRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZsytrsRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.ZsytrsRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZsytrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("ZsytrsRook", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.ZsytrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsRook", err)

		//       ZsyconRook
		*srnamt = "ZsyconRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsyconRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZsyconRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsyconRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZsyconRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsyconRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZsyconRook", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.ZsyconRook(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -anrm, w)
		chkxer2("ZsyconRook", err)

	} else if c2 == "sk" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//       ZsytrfRk
		*srnamt = "ZsytrfRk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsytrfRk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("ZsytrfRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsytrfRk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("ZsytrfRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsytrfRk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 4)
		chkxer2("ZsytrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZsytrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("ZsytrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZsytrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("ZsytrfRk", err)

		//       Zsytf2Rk
		*srnamt = "Zsytf2Rk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytf2Rk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zsytf2Rk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytf2Rk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zsytf2Rk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytf2Rk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zsytf2Rk", err)

		//       Zsytri3
		*srnamt = "Zsytri3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytri3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zsytri3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytri3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zsytri3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytri3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zsytri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=0, lwkopt=8, lquery=false")
		_, err = golapack.Zsytri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("Zsytri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=-2, lwkopt=8, lquery=false")
		_, err = golapack.Zsytri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("Zsytri3", err)

		//       Zsytri3x
		*srnamt = "Zsytri3x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsytri3x('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri3x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsytri3x(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri3x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsytri3x(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zsytri3x", err)

		//       Zsytrs3
		*srnamt = "Zsytrs3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zsytrs3('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zsytrs3(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs3", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zsytrs3(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zsytrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(2, opts))
		chkxer2("Zsytrs3", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zsytrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zsytrs3", err)

		//       Zsycon3
		*srnamt = "Zsycon3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsycon3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zsycon3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsycon3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zsycon3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zsycon3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zsycon3", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zsycon3(Upper, 1, a.Off(0, 0).UpdateRows(1), e, &ip, -1.0, w)
		chkxer2("Zsycon3", err)

	} else if c2 == "sp" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite packed matrix with patrial
		//        (Bunch-Kaufman) pivoting.
		//
		//       Zsptrf
		*srnamt = "Zsptrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsptrf('/', 0, a.Off(0, 0).CVector(), &ip)
		chkxer2("Zsptrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsptrf(Upper, -1, a.Off(0, 0).CVector(), &ip)
		chkxer2("Zsptrf", err)

		//       Zsptri
		*srnamt = "Zsptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zsptri('/', 0, a.Off(0, 0).CVector(), &ip, w)
		chkxer2("Zsptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zsptri(Upper, -1, a.Off(0, 0).CVector(), &ip, w)
		chkxer2("Zsptri", err)

		//       Zsptrs
		*srnamt = "Zsptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zsptrs('/', 0, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zsptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zsptrs(Upper, -1, 0, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zsptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zsptrs(Upper, 0, -1, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zsptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Row=1, n=2")
		err = golapack.Zsptrs(Upper, 2, 1, a.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts))
		chkxer2("Zsptrs", err)

		//       Zsprfs
		*srnamt = "Zsprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zsprfs('/', 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zsprfs(Upper, -1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zsprfs(Upper, 0, -1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zsprfs(Upper, 2, 1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zsprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zsprfs(Upper, 2, 1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zsprfs", err)

		//       Zspcon
		*srnamt = "Zspcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zspcon('/', 0, a.Off(0, 0).CVector(), &ip, anrm, w)
		chkxer2("Zspcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zspcon(Upper, -1, a.Off(0, 0).CVector(), &ip, anrm, w)
		chkxer2("Zspcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zspcon(Upper, 1, a.Off(0, 0).CVector(), &ip, -anrm, w)
		chkxer2("Zspcon", err)

	} else if c2 == "sa" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//       ZsytrfAa
		*srnamt = "ZsytrfAa"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZsytrfAa('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZsytrfAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZsytrfAa(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZsytrfAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZsytrfAa(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("ZsytrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=0, n=0, lquery=false")
		err = golapack.ZsytrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("ZsytrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=-2, n=0, lquery=false")
		err = golapack.ZsytrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("ZsytrfAa", err)

		//       ZsytrsAa
		*srnamt = "ZsytrsAa"
		// *errt = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsytrsAa('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsytrsAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsytrsAa(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsytrsAa", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZsytrsAa(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsytrsAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsytrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZsytrsAa", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZsytrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZsytrsAa", err)

	} else if c2 == "S2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//       ZsytrfAa2stage
		*srnamt = "ZsytrfAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZsytrfAa2stage('/', 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, w, 1)
		chkxer2("ZsytrfAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZsytrfAa2stage(Upper, -1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, w, 1)
		chkxer2("ZsytrfAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 2, &ip, &ip, w, 1)
		chkxer2("ZsytrfAa2stage", err)
		*errt = fmt.Errorf("ltb < 4*n && !tquery: ltb=1, n=2, tquery=1")
		_, err = golapack.ZsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 1, &ip, &ip, w, 1)
		chkxer2("ZsytrfAa2stage", err)
		*errt = fmt.Errorf("lwork < n && !wquery: lwork=0, n=2, wquery=false")
		_, err = golapack.ZsytrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 8, &ip, &ip, w, 0)
		chkxer2("ZsytrfAa2stage", err)

		//        CHETRS_AA_2STAGE
		*srnamt = "ZsytrsAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZsytrsAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZsytrsAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.ZsytrsAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAa2stage", err)
		*errt = fmt.Errorf("ltb < (4 * n): ltb=1, n=2")
		err = golapack.ZsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.ZsytrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.Off(0, 0).CVector(), 8, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZsytrsAaStage", err)

	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
