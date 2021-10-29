package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// derrtz tests the error exits for STZRZF.
func derrtz(path string, t *testing.T) {
	var err error

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	a := mf(2, 2, opts)
	tau := vf(2)
	w := vf(2)

	c2 := path[1:3]
	a.Set(0, 0, 1.)
	a.Set(0, 1, 2.)
	a.Set(1, 1, 3.)
	a.Set(1, 0, 4.)
	w.Set(0, 0.0)
	w.Set(1, 0.0)
	(*ok) = true

	if c2 == "tz" {
		//        Test error exits for the trapezoidal routines.
		//
		//        Dtzrzf
		*srnamt = "Dtzrzf"
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Dtzrzf(-1, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dtzrzf", err)
		*errt = fmt.Errorf("n < m: m=1, n=0")
		err = golapack.Dtzrzf(1, 0, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dtzrzf", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Dtzrzf(2, 2, a.Off(0, 0).UpdateRows(1), tau, w, 1)
		chkxer2("Dtzrzf", err)
		*errt = fmt.Errorf("lwork < lwkmin && !lquery: lwork=0, lwkmin=1, lquery=false")
		err = golapack.Dtzrzf(2, 2, a.Off(0, 0).UpdateRows(2), tau, w, 0)
		chkxer2("Dtzrzf", err)
		*errt = fmt.Errorf("lwork < lwkmin && !lquery: lwork=1, lwkmin=2, lquery=false")
		err = golapack.Dtzrzf(2, 3, a.Off(0, 0).UpdateRows(2), tau, w, 1)
		chkxer2("Dtzrzf", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
