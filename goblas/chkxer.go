package goblas

import (
	"fmt"
)

// Chkxer ...
func Chkxer(srnamt string, err error) {
	errt := common.infoc.errt
	ok := &common.infoc.ok

	if err.Error() != errt.Error() {
		fmt.Printf(" *** Illegal value\n got: %v\n want: %v\n not detected by %6s ***\n", err, errt, srnamt[1:])
		*ok = false
	}
}
