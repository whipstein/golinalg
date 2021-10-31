package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrps tests the error exits for the COMPLEX routines
// for Zpstrf.
func zerrps(path string, t *testing.T) {
	var i, j, nmax int
	var err error

	nmax = 4
	rwork := vf(2 * nmax)
	piv := make([]int, 4)
	a := cmf(4, 4, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))

		}
		piv[j-1] = j
		rwork.Set(j-1, 0.)
		rwork.Set(nmax+j-1, 0.)

	}
	*ok = true

	//        Test error exits of the routines that use the Cholesky
	//        decomposition of an Hermitian positive semidefinite matrix.
	//
	//        Zpstrf
	*srnamt = "Zpstrf"
	*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
	_, _, err = golapack.Zpstrf('/', 0, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstrf", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Zpstrf(Upper, -1, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstrf", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	_, _, err = golapack.Zpstrf(Upper, 2, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstrf", err)

	//        Zpstf2
	*srnamt = "Zpstf2"
	*errt = fmt.Errorf("!upper && uplo != Lower: uplo=Unrecognized: /")
	_, _, err = golapack.Zpstf2('/', 0, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstf2", err)
	*errt = fmt.Errorf("n < 0: n=-1")
	_, _, err = golapack.Zpstf2(Upper, -1, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstf2", err)
	*errt = fmt.Errorf("a.Rows < max(1, n): a.Rows=1, n=2")
	_, _, err = golapack.Zpstf2(Upper, 2, a.Off(0, 0).UpdateRows(1), &piv, -1., rwork)
	chkxer2("Zpstf2", err)

	//     Print a summary line.
	// alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
