package gltest

import "fmt"

// Xerbla is an error handler for the LAPACK routines.
// It is called by an LAPACK routine if an input parameter has an
// invalid value.  A message is printed and execution stops.
//
// Installers may consider modifying the STOP statement in order to
// call system-specific exception-handling facilities.
func Xerbla(srname string, info int) {
	infot := Common.Infoc.Infot
	ok := &Common.Infoc.Ok
	lerr := &Common.Infoc.Lerr
	srnamt := Common.Srnamc.Srnamt

	(*lerr) = true
	if info != infot {
		if infot != 0 {
			fmt.Printf(" *** XERBLA was called from %s with INFO = %6d instead of %2d ***\n", srnamt, info, infot)
		} else {
			fmt.Printf(" *** On entry to %s parameter number %6d had an illegal value ***\n", srname, info)
		}
		(*ok) = false
	}
	if string(srname) != srnamt && srnamt != "" {
		fmt.Printf(" *** XERBLA was called with SRNAME = %s instead of %6s ***\n", srname, srnamt)
		(*ok) = false
	}
}

func Xerbla2(srname string, err error) {
	errt := Common.Infoc.Errt
	ok := &Common.Infoc.Ok
	lerr := &Common.Infoc.Lerr
	srnamt := Common.Srnamc.Srnamt

	(*lerr) = true
	if err != nil && errt == nil {
		fmt.Printf(" *** Illegal value\n got:  %v\n want: %v\n not detected by %6s ***\n", err, errt, srname)
		(*ok) = false
	}
	if srname != srnamt && srnamt != "" {
		fmt.Printf(" *** XERBLA was called with SRNAME = %s instead of %6s ***\n", srname, srnamt)
		(*ok) = false
	}
}
