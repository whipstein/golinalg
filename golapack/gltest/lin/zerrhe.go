package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrhe tests the error exits for the COMPLEX*16 routines
// for Hermitian indefinite matrices.
func zerrhe(path string, t *testing.T) {
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

	if c2 == "he" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//        Zhetrf
		*srnamt = "Zhetrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetrf('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zhetrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetrf(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zhetrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetrf(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("Zhetrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.Zhetrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("Zhetrf", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.Zhetrf(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("Zhetrf", err)

		//        Zhetf2
		*srnamt = "Zhetf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetf2('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetf2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetf2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2", err)

		//        Zhetri
		*srnamt = "Zhetri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetri('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zhetri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetri(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zhetri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetri(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("Zhetri", err)

		//        Zhetri2
		*srnamt = "Zhetri2"
		// *errt = fmt.Errorf("lwork < minsize && !lquery: lwork=%v, minsize=%v, lquery=%v", lwork, minsize, lquery)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetri2('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zhetri2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetri2(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zhetri2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetri2(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("Zhetri2", err)

		//        Zhetri2x
		*srnamt = "Zhetri2x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetri2x('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri2x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetri2x(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri2x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetri2x(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri2x", err)

		//        Zhetrs
		*srnamt = "Zhetrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zhetrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zhetrs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zhetrs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zhetrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("Zhetrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zhetrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs", err)

		//        Zhetrfs
		*srnamt = "Zherfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zherfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zherfs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zherfs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zherfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Zherfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zherfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zherfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zherfs", err)

		//        Zhecon
		*srnamt = "Zhecon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhecon('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zhecon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhecon(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zhecon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhecon(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("Zhecon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zhecon(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -anrm, w)
		chkxer2("Zhecon", err)

	} else if c2 == "hr" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with rook
		//        (bounded Bunch-Kaufman) diagonal pivoting method.
		//
		//        ZhetrfRook
		*srnamt = "ZhetrfRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhetrfRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZhetrfRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhetrfRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZhetrfRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhetrfRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("ZhetrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZhetrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("ZhetrfRook", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZhetrfRook(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("ZhetrfRook", err)

		//        Zhetf2Rook
		*srnamt = "Zhetf2Rook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetf2Rook('/', 0, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2Rook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetf2Rook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2Rook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetf2Rook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip)
		chkxer2("Zhetf2Rook", err)

		//        ZhetriRook
		*srnamt = "ZhetriRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhetriRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZhetriRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhetriRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZhetriRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhetriRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w)
		chkxer2("ZhetriRook", err)

		//        ZhetrsRook
		*srnamt = "ZhetrsRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZhetrsRook('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZhetrsRook(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsRook", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.ZhetrsRook(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZhetrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts))
		chkxer2("ZhetrsRook", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.ZhetrsRook(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsRook", err)

		//        ZheconRook
		*srnamt = "ZheconRook"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZheconRook('/', 0, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZheconRook", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZheconRook(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZheconRook", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZheconRook(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, anrm, w)
		chkxer2("ZheconRook", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.ZheconRook(Upper, 1, a.Off(0, 0).UpdateRows(1), &ip, -anrm, w)
		chkxer2("ZheconRook", err)

	} else if c2 == "hk" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with rook
		//        (bounded Bunch-Kaufman) pivoting with the new storage
		//        format for factors L ( or U) and D.
		//
		//        L (or U) is stored in A, diagonal of D is stored on the
		//        diagonal of A, subdiagonal of D is stored in a separate array E.
		//
		//        ZhetrfRk
		*srnamt = "ZhetrfRk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhetrfRk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("ZhetrfRk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhetrfRk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("ZhetrfRk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhetrfRk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 4)
		chkxer2("ZhetrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=0, lquery=false")
		_, err = golapack.ZhetrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("ZhetrfRk", err)
		*errt = fmt.Errorf("lwork < 1 && !lquery: lwork=-2, lquery=false")
		_, err = golapack.ZhetrfRk(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("ZhetrfRk", err)

		//        Zhetf2Rk
		*srnamt = "Zhetf2Rk"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetf2Rk('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zhetf2Rk", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetf2Rk(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zhetf2Rk", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetf2Rk(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip)
		chkxer2("Zhetf2Rk", err)

		//        Zhetri3
		*srnamt = "Zhetri3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetri3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zhetri3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetri3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zhetri3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetri3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w, 1)
		chkxer2("Zhetri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=0, lwkopt=8, lquery=false")
		_, err = golapack.Zhetri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, 0)
		chkxer2("Zhetri3", err)
		*errt = fmt.Errorf("lwork < lwkopt && !lquery: lwork=-2, lwkopt=8, lquery=false")
		_, err = golapack.Zhetri3(Upper, 0, a.Off(0, 0).UpdateRows(1), e, &ip, w, -2)
		chkxer2("Zhetri3", err)

		//        Zhetri3x
		*srnamt = "Zhetri3x"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhetri3x('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri3x", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhetri3x(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri3x", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhetri3x(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, w.CMatrix(1, opts), 1)
		chkxer2("Zhetri3x", err)

		//        Zhetrs3
		*srnamt = "Zhetrs3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zhetrs3('/', 0, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zhetrs3(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs3", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zhetrs3(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zhetrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), e, &ip, b.CMatrix(2, opts))
		chkxer2("Zhetrs3", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zhetrs3(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), e, &ip, b.CMatrix(1, opts))
		chkxer2("Zhetrs3", err)

		//        Zhecon3
		*srnamt = "Zhecon3"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhecon3('/', 0, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zhecon3", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhecon3(Upper, -1, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zhecon3", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zhecon3(Upper, 2, a.Off(0, 0).UpdateRows(1), e, &ip, anrm, w)
		chkxer2("Zhecon3", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zhecon3(Upper, 1, a.Off(0, 0).UpdateRows(1), e, &ip, -1.0, w)
		chkxer2("Zhecon3", err)

		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite matrix with Aasen's algorithm.
	} else if c2 == "ha" {
		//        ZhetrfAa
		*srnamt = "ZhetrfAa"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZhetrfAa('/', 0, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZhetrfAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZhetrfAa(Upper, -1, a.Off(0, 0).UpdateRows(1), &ip, w, 1)
		chkxer2("ZhetrfAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZhetrfAa(Upper, 2, a.Off(0, 0).UpdateRows(1), &ip, w, 4)
		chkxer2("ZhetrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=0, n=0, lquery=false")
		err = golapack.ZhetrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, 0)
		chkxer2("ZhetrfAa", err)
		*errt = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=-2, n=0, lquery=false")
		err = golapack.ZhetrfAa(Upper, 0, a.Off(0, 0).UpdateRows(1), &ip, w, -2)
		chkxer2("ZhetrfAa", err)

		//        ZhetrsAa
		*srnamt = "ZhetrsAa"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhetrsAa('/', 0, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhetrsAa(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		_, err = golapack.ZhetrsAa(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhetrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(2, opts), w, 1)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		_, err = golapack.ZhetrsAa(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), &ip, b.CMatrix(1, opts), w, 1)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=0, n=0, lquery=false")
		_, err = golapack.ZhetrsAa(Upper, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, 0)
		chkxer2("ZhetrsAa", err)
		*errt = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=-2, n=0, lquery=false")
		_, err = golapack.ZhetrsAa(Upper, 0, 1, a.Off(0, 0).UpdateRows(1), &ip, b.CMatrix(1, opts), w, -2)
		chkxer2("ZhetrsAa", err)

	} else if c2 == "s2" {
		//        Test error exits of the routines that use factorization
		//        of a symmetric indefinite matrix with Aasen's algorithm.
		//
		//        ZhetrfAa2stage
		*srnamt = "ZhetrfAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.ZhetrfAa2stage('/', 0, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, w, 1)
		chkxer2("ZhetrfAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.ZhetrfAa2stage(Upper, -1, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, w, 1)
		chkxer2("ZhetrfAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.ZhetrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 2, &ip, &ip, w, 1)
		chkxer2("ZhetrfAa2stage", err)
		*errt = fmt.Errorf("ltb < 4*n && !tquery: ltb=1, n=2, tquery=false")
		_, err = golapack.ZhetrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), a.CVector(0, 0), 1, &ip, &ip, w, 1)
		chkxer2("ZhetrfAa2stage", err)
		*errt = fmt.Errorf("lwork < n && !wquery: lwork=0, n=2, wquery=false")
		_, err = golapack.ZhetrfAa2stage(Upper, 2, a.Off(0, 0).UpdateRows(2), a.CVector(0, 0), 8, &ip, &ip, w, 0)
		chkxer2("ZhetrfAa2stage", err)

		//        ZhetrsAa2stage
		*srnamt = "ZhetrsAa2stage"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.ZhetrsAa2stage('/', 0, 0, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.ZhetrsAa2stage(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.ZhetrsAa2stage(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.ZhetrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), a.CVector(0, 0), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)
		*errt = fmt.Errorf("ltb < (4 * n): ltb=1, n=2")
		err = golapack.ZhetrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.CVector(0, 0), 1, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.ZhetrsAa2stage(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), a.CVector(0, 0), 8, &ip, &ip, b.CMatrix(1, opts))
		chkxer2("ZhetrsAa2stage", err)

	} else if c2 == "hp" {
		//        Test error exits of the routines that use factorization
		//        of a Hermitian indefinite packed matrix with patrial
		//        (Bunch-Kaufman) diagonal pivoting method.
		//
		//        Zhptrf
		*srnamt = "Zhptrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhptrf('/', 0, a.CVector(0, 0), &ip)
		chkxer2("Zhptrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhptrf(Upper, -1, a.CVector(0, 0), &ip)
		chkxer2("Zhptrf", err)

		//        Zhptri
		*srnamt = "Zhptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhptri('/', 0, a.CVector(0, 0), &ip, w)
		chkxer2("Zhptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhptri(Upper, -1, a.CVector(0, 0), &ip, w)
		chkxer2("Zhptri", err)

		//        Zhptrs
		*srnamt = "Zhptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zhptrs('/', 0, 0, a.CVector(0, 0), &ip, b.CMatrix(1, opts))
		chkxer2("Zhptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zhptrs(Upper, -1, 0, a.CVector(0, 0), &ip, b.CMatrix(1, opts))
		chkxer2("Zhptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zhptrs(Upper, 0, -1, a.CVector(0, 0), &ip, b.CMatrix(1, opts))
		chkxer2("Zhptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zhptrs(Upper, 2, 1, a.CVector(0, 0), &ip, b.CMatrix(1, opts))
		chkxer2("Zhptrs", err)

		//        Zhprfs
		*srnamt = "Zhprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zhprfs('/', 0, 0, a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zhprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zhprfs(Upper, -1, 0, a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zhprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zhprfs(Upper, 0, -1, a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zhprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zhprfs(Upper, 2, 1, a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zhprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zhprfs(Upper, 2, 1, a.CVector(0, 0), af.CVector(0, 0), &ip, b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zhprfs", err)

		//        Zhpcon
		*srnamt = "Zhpcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zhpcon('/', 0, a.CVector(0, 0), &ip, anrm, w)
		chkxer2("Zhpcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zhpcon(Upper, -1, a.CVector(0, 0), &ip, anrm, w)
		chkxer2("Zhpcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zhpcon(Upper, 1, a.CVector(0, 0), &ip, -anrm, w)
		chkxer2("Zhpcon", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
