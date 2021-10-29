package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// chkxer ...
func chkxer(srnamt string, info int, lerr, ok *bool, t *testing.T) {
	infot := gltest.Common.Infoc.Infot

	if abs(info) != abs(infot) {
		t.Fail()
		fmt.Printf(" *** Illegal value of parameter number %2d not detected by %6s ***\n", info, srnamt[1:])
		*ok = false
	}
	*lerr = false
}

func chkxer2(srnamt string, err error) {
	errt := gltest.Common.Infoc.Errt
	ok := &gltest.Common.Infoc.Ok

	if err == nil || err.Error() != errt.Error() {
		fmt.Printf(" *** Illegal value\n got:  %v\n want: %v\n not detected by %s ***\n", err, errt, srnamt)
		*ok = false
	}
}
