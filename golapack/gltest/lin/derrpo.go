package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrpo tests the error exits for the DOUBLE PRECISION routines
// for symmetric positive definite matrices.
func derrpo(path string, t *testing.T) {
	var anrm float64
	var i, j, nmax int
	var err error

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
	w := vf(3 * nmax)
	x := mf(nmax, 1, opts)
	r1 := vf(4)
	r2 := vf(4)

	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.SetIdx(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.SetIdx(j-1, 0.)
		iw[j-1] = j
	}
	(*ok) = true

	if c2 == "po" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite matrix.
		//
		//        Dpotrf
		*srnamt = "Dpotrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpotrf('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpotrf(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrf", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dpotrf(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrf", err)

		//        Dpotf2
		*srnamt = "Dpotf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpotf2('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpotf2(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotf2", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dpotf2(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotf2", err)

		//        Dpotri
		*srnamt = "Dpotri"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpotri('/', 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpotri(Upper, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotri", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dpotri(Upper, 2, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotri", err)

		//        Dpotrs
		*srnamt = "Dpotrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dpotrs('/', 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpotrs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpotrs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dpotrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2))
		chkxer2("Dpotrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpotrs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpotrs", err)

		//        Dporfs
		*srnamt = "Dporfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dporfs('/', 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dporfs(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dporfs(Upper, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		err = golapack.Dporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("af.Rows < max(1, n): af.Rows=1, n=2")
		err = golapack.Dporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dporfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dporfs(Upper, 2, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dporfs", err)

		//        Dpocon
		*srnamt = "Dpocon"
		// *errt = fmt.Errorf("anorm < zero: anorm=%v", anorm)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpocon('/', 0, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpocon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpocon(Upper, -1, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpocon", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, err = golapack.Dpocon(Upper, 2, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpocon", err)

		//        Dpoequ
		*srnamt = "Dpoequ"
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dpoequ(-1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpoequ", err)
		*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
		_, _, _, err = golapack.Dpoequ(2, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpoequ", err)

	} else if c2 == "pp" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite packed matrix.
		//
		//        Dpptrf
		*srnamt = "Dpptrf"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpptrf('/', 0, ap)
		chkxer2("Dpptrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpptrf(Upper, -1, ap)
		chkxer2("Dpptrf", err)

		//        Dpptri
		*srnamt = "Dpptri"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpptri('/', 0, ap)
		chkxer2("Dpptri", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpptri(Upper, -1, ap)
		chkxer2("Dpptri", err)

		//        Dpptrs
		*srnamt = "Dpptrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dpptrs('/', 0, 0, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpptrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpptrs(Upper, -1, 0, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpptrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpptrs(Upper, 0, -1, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpptrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpptrs(Upper, 2, 1, ap, b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpptrs", err)

		//        Dpprfs
		*srnamt = "Dpprfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dpprfs('/', 0, 0, ap, afp, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpprfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpprfs(Upper, -1, 0, ap, afp, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpprfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpprfs(Upper, 0, -1, ap, afp, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpprfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpprfs(Upper, 2, 1, ap, afp, b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpprfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dpprfs(Upper, 2, 1, ap, afp, b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpprfs", err)

		//        Dppcon
		*srnamt = "Dppcon"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dppcon('/', 0, ap, anrm, w, &iw)
		chkxer2("Dppcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dppcon(Upper, -1, ap, anrm, w, &iw)
		chkxer2("Dppcon", err)

		//        Dppequ
		*srnamt = "Dppequ"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Dppequ('/', 0, ap, r1)
		chkxer2("Dppequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dppequ(Upper, -1, ap, r1)
		chkxer2("Dppequ", err)

	} else if c2 == "pb" {
		//        Test error exits of the routines that use the Cholesky
		//        decomposition of a symmetric positive definite band matrix.
		//
		//        Dpbtrf
		*srnamt = "Dpbtrf"
		*errt = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpbtrf('/', 0, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrf", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpbtrf(Upper, -1, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrf", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dpbtrf(Upper, 1, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrf", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dpbtrf(Upper, 2, 1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrf", err)

		//        Dpbtf2
		*srnamt = "Dpbtf2"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpbtf2('/', 0, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtf2", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpbtf2(Upper, -1, 0, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtf2", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dpbtf2(Upper, 1, -1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtf2", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dpbtf2(Upper, 2, 1, a.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtf2", err)

		//        Dpbtrs
		*srnamt = "Dpbtrs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dpbtrs('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpbtrs(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Dpbtrs(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpbtrs(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Dpbtrs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpbtrs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1))
		chkxer2("Dpbtrs", err)

		//        Dpbrfs
		*srnamt = "Dpbrfs"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		err = golapack.Dpbrfs('/', 0, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		err = golapack.Dpbrfs(Upper, -1, 0, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		err = golapack.Dpbrfs(Upper, 1, -1, 0, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("nrhs < 0: nrhs=-1")
		err = golapack.Dpbrfs(Upper, 0, 0, -1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		err = golapack.Dpbrfs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(2), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("afb.Rows < kd+1: afb.Rows=1, kd=1")
		err = golapack.Dpbrfs(Upper, 2, 1, 1, a.Off(0, 0).UpdateRows(2), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("b.Rows < max(1, n): b.Rows=1, n=2")
		err = golapack.Dpbrfs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(1), x.Off(0, 0).UpdateRows(2), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)
		*errt = fmt.Errorf("x.Rows < max(1, n): x.Rows=1, n=2")
		err = golapack.Dpbrfs(Upper, 2, 0, 1, a.Off(0, 0).UpdateRows(1), af.Off(0, 0).UpdateRows(1), b.Off(0, 0).UpdateRows(2), x.Off(0, 0).UpdateRows(1), r1, r2, w, &iw)
		chkxer2("Dpbrfs", err)

		//        Dpbcon
		*srnamt = "Dpbcon"
		// *errt = fmt.Errorf("anorm < zero: anorm=%v", anorm)
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, err = golapack.Dpbcon('/', 0, 0, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpbcon", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, err = golapack.Dpbcon(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpbcon", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, err = golapack.Dpbcon(Upper, 1, -1, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpbcon", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, err = golapack.Dpbcon(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), anrm, w, &iw)
		chkxer2("Dpbcon", err)

		//        Dpbequ
		*srnamt = "Dpbequ"
		*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
		_, _, _, err = golapack.Dpbequ('/', 0, 0, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpbequ", err)
		*errt = fmt.Errorf("n < 0: n=-1")
		_, _, _, err = golapack.Dpbequ(Upper, -1, 0, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpbequ", err)
		*errt = fmt.Errorf("kd < 0: kd=-1")
		_, _, _, err = golapack.Dpbequ(Upper, 1, -1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpbequ", err)
		*errt = fmt.Errorf("ab.Rows < kd+1: ab.Rows=1, kd=1")
		_, _, _, err = golapack.Dpbequ(Upper, 2, 1, a.Off(0, 0).UpdateRows(1), r1)
		chkxer2("Dpbequ", err)
	}

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
