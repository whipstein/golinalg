package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrps tests the error exits for the DOUBLE PRECISION routines
// for Dpstrf.
func derrps(path string, t *testing.T) {
	var i, j, nmax int
	var err error

	piv := make([]int, 4)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4

	a := mf(4, 4, opts)
	work := vf(2 * nmax)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))

		}
		piv[j-1] = j
		work.Set(j-1, 0.)
		work.Set(nmax+j-1, 0.)

	}
	(*ok) = true

	//        Test error exits of the routines that use the Cholesky
	//        decomposition of a symmetric positive semidefinite matrix.
	//
	//        Dpstrf
	*srnamt = "Dpstrf"
	*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
	_, _, err = golapack.Dpstrf('/', 0, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstrf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Dpstrf(Upper, -1, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstrf", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	_, _, err = golapack.Dpstrf(Upper, 2, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstrf", err)

	//        Dpstf2
	*srnamt = "Dpstf2"
	*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
	_, _, err = golapack.Dpstf2('/', 0, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstf2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Dpstf2(Upper, -1, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstf2", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	_, _, err = golapack.Dpstf2(Upper, 2, a.Off(0, 0).UpdateRows(1), &piv, -1.0, work)
	chkxer2("Dpstf2", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
