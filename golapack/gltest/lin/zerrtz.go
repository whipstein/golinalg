package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// zerrtz tests the error exits for Ztzrzf.
func zerrtz(path string, t *testing.T) {
	var err error

	tau := cvf(2)
	w := cvf(2)
	a := cmf(2, 2, opts)

	errt := &gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok
	srnamt := &gltest.Common.Srnamc.Srnamt

	c2 := path[1:3]
	a.Set(0, 0, complex(1, -1.))
	a.Set(0, 1, complex(2, -2.))
	a.Set(1, 1, complex(3, -3.))
	a.Set(1, 0, complex(4, -4.))
	w.Set(0, complex(0, 0))
	w.Set(1, complex(0, 0))
	(*ok) = true

	//     Test error exits for the trapezoidal routines.
	if c2 == "tz" {
		//        Ztzrzf
		*srnamt = "Ztzrzf"
		*errt = fmt.Errorf("m < 0: m=-1")
		err = golapack.Ztzrzf(-1, 0, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Ztzrzf", err)
		*errt = fmt.Errorf("n < m: m=1, n=0")
		err = golapack.Ztzrzf(1, 0, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Ztzrzf", err)
		*errt = fmt.Errorf("a.Rows < max(1, m): a.Rows=1, m=2")
		err = golapack.Ztzrzf(2, 2, a.Off(0, 0).UpdateRows(1), tau, w.Off(0, 1), 1)
		chkxer2("Ztzrzf", err)
		*errt = fmt.Errorf("lwork < lwkmin && !lquery: lwork=0, lwkmin=1, lquery=false")
		err = golapack.Ztzrzf(2, 2, a.Off(0, 0).UpdateRows(2), tau, w.Off(0, 0), 0)
		chkxer2("Ztzrzf", err)
		*errt = fmt.Errorf("lwork < lwkmin && !lquery: lwork=1, lwkmin=2, lquery=false")
		err = golapack.Ztzrzf(2, 3, a.Off(0, 0).UpdateRows(2), tau, w.Off(0, 1), 1)
		chkxer2("Ztzrzf", err)
	}

	//     Print a summary line.
	alaesm(path, *ok)

	if !(*ok) {
		t.Fail()
	}
}
