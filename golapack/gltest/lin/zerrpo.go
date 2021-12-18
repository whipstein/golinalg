package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrpo tests the error exits for the COMPLEX*16 routines
// for Hermitian positive definite matrices.
func zerrpo(path string, t *testing.T) {
	var anrm float64
	var i, j, nmax int
	var err error

	nmax = 4
	b := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
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
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	anrm = 1.
	(*ok) = true

	//     Test error exits of the routines that use the Cholesky
	//     decomposition of a Hermitian positive definite matrix.
	if c2 == "po" {
		//        Zpotrf
		*srnamt = "Zpotrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpotrf('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpotrf(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zpotrf(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotrf", err)

		//        Zpotf2
		*srnamt = "Zpotf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpotf2('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpotf2(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zpotf2(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotf2", err)

		//        Zpotri
		*srnamt = "Zpotri"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpotri('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpotri(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zpotri(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpotri", err)

		//        Zpotrs
		*srnamt = "Zpotrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpotrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpotrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpotrs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpotrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpotrs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpotrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zpotrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts))
		chkxer2("Zpotrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpotrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), b.CMatrix(1, opts))
		chkxer2("Zpotrs", err)

		//        Zporfs
		*srnamt = "Zporfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zporfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zporfs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zporfs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Zporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Zporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zporfs", err)

		//        Zpocon
		*srnamt = "Zpocon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpocon('/', 0, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpocon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpocon(Upper, -1, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpocon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Zpocon(Upper, 2, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpocon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zpocon(Upper, 1, a.Off(0, 0).UpdateRows(1), -anrm, w, r)
		chkxer2("Zpocon", err)

		//        Zpoequ
		*srnamt = "Zpoequ"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zpoequ(-1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpoequ", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Zpoequ(2, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpoequ", err)

		//     Test error exits of the routines that use the Cholesky
		//     decomposition of a Hermitian positive definite packed matrix.
	} else if c2 == "pp" {
		//        Zpptrf
		*srnamt = "Zpptrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpptrf('/', 0, a.CVector())
		chkxer2("Zpptrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpptrf(Upper, -1, a.CVector())
		chkxer2("Zpptrf", err)

		//        Zpptri
		*srnamt = "Zpptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpptri('/', 0, a.CVector())
		chkxer2("Zpptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpptri(Upper, -1, a.CVector())
		chkxer2("Zpptri", err)

		//        Zpptrs
		*srnamt = "Zpptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpptrs('/', 0, 0, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zpptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpptrs(Upper, -1, 0, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zpptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpptrs(Upper, 0, -1, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zpptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpptrs(Upper, 2, 1, a.Off(0, 0).CVector(), b.CMatrix(1, opts))
		chkxer2("Zpptrs", err)

		//        Zpprfs
		*srnamt = "Zpprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpprfs('/', 0, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpprfs(Upper, -1, 0, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpprfs(Upper, 0, -1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpprfs(Upper, 2, 1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zpprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zpprfs(Upper, 2, 1, a.Off(0, 0).CVector(), af.Off(0, 0).CVector(), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpprfs", err)

		//        Zppcon
		*srnamt = "Zppcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zppcon('/', 0, a.Off(0, 0).CVector(), anrm, w, r)
		chkxer2("Zppcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zppcon(Upper, -1, a.Off(0, 0).CVector(), anrm, w, r)
		chkxer2("Zppcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zppcon(Upper, 1, a.Off(0, 0).CVector(), -anrm, w, r)
		chkxer2("Zppcon", err)

		//        Zppequ
		*srnamt = "Zppequ"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Zppequ('/', 0, a.Off(0, 0).CVector(), r1)
		chkxer2("Zppequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zppequ(Upper, -1, a.Off(0, 0).CVector(), r1)
		chkxer2("Zppequ", err)

		//     Test error exits of the routines that use the Cholesky
		//     decomposition of a Hermitian positive definite band matrix.
	} else if c2 == "pb" {
		//        Zpbtrf
		*srnamt = "Zpbtrf"
		*errt = fmt.Errorf("(uplo != Upper) && (uplo != Lower): uplo=Unrecognized: /")
		_, err = golapack.Zpbtrf('/', 0, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpbtrf(Upper, -1, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtrf", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Zpbtrf(Upper, 1, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtrf", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Zpbtrf(Upper, 2, 1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtrf", err)

		//        Zpbtf2
		*srnamt = "Zpbtf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpbtf2('/', 0, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpbtf2(Upper, -1, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtf2", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Zpbtf2(Upper, 1, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtf2", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Zpbtf2(Upper, 2, 1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Zpbtf2", err)

		//        Zpbtrs
		*srnamt = "Zpbtrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpbtrs('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpbtrs(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Zpbtrs(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpbtrs(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Zpbtrs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpbtrs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts))
		chkxer2("Zpbtrs", err)

		//        Zpbrfs
		*srnamt = "Zpbrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Zpbrfs('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Zpbrfs(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Zpbrfs(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Zpbrfs(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Zpbrfs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("afb.Rows < kd+1: afb.Rows=1, kd=1")
		err = golapack.Zpbrfs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Zpbrfs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(1, opts), x.CMatrix(2, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Zpbrfs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.CMatrix(2, opts), x.CMatrix(1, opts), r1, r2, w, r)
		chkxer2("Zpbrfs", err)

		//        Zpbcon
		*srnamt = "Zpbcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Zpbcon('/', 0, 0, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Zpbcon(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpbcon", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Zpbcon(Upper, 1, -1, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpbcon", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Zpbcon(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), anrm, w, r)
		chkxer2("Zpbcon", err)
		*errt = fmt.Errorf("anorm < zero: anorm=-1")
		_, err = golapack.Zpbcon(Upper, 1, 0, a.Off(0, 0).UpdateRows(1), -anrm, w, r)
		chkxer2("Zpbcon", err)

		//        Zpbequ
		*srnamt = "Zpbequ"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Zpbequ('/', 0, 0, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpbequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Zpbequ(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpbequ", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, _, _, err = golapack.Zpbequ(Upper, 1, -1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpbequ", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, _, _, err = golapack.Zpbequ(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Zpbequ", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
