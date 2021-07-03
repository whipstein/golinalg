package goblas

import (
	"log"
)

// Xerbla is an error handler for the LAPACK routines.
// It is called by an LAPACK routine if an input parameter has an
// invalid value.  A message is printed and execution stops.
//
// Installers may consider modifying the STOP statement in order to
// call system-specific exception-handling facilities.
func Xerbla(srname []byte, info int) {
	log.Panicf(" ** On entry to %s parameter number %2d had an illegal value\n", srname[1:], info)
}

func Xerbla2(srname []byte, err error) {
	lerr := &common.infoc.lerr

	if *lerr {
		return
	}
	log.Panicf(" ** On entry to %s\n%v\nhad an illegal value\n", srname[1:], err)
}
