package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
)

// Chkxer ...
func Chkxer(srnamt string, info *int, lerr, ok *bool, t *testing.T) {
	infot := &gltest.Common.Infoc.Infot

	if absint(*info) != absint(*infot) {
		t.Fail()
		fmt.Printf(" *** Illegal value of parameter number %2d not detected by %6s ***\n", *info, srnamt[1:])
		*ok = false
	}
	*lerr = false
}
