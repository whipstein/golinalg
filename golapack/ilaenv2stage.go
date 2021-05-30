package golapack

// Ilaenv2stage is called from the LAPACK routines to choose problem-dependent
// parameters for the local environment.  See ISPEC for a description of
// the parameters.
// It sets problem and machine dependent parameters useful for *_2STAGE and
// related subroutines.
//
// ILAENV2STAGE returns an INTEGER
// if ILAENV2STAGE >= 0: ILAENV2STAGE returns the value of the parameter
//                       specified by ISPEC
// if ILAENV2STAGE < 0:  if ILAENV2STAGE = -k, the k-th argument had an
//                       illegal value.
//
// This version provides a set of parameters which should give good,
// but not optimal, performance on many of the currently available
// computers for the 2-stage solvers. Users are encouraged to modify this
// subroutine to set the tuning parameters for their particular machine using
// the option and problem size information in the arguments.
//
// This routine will not function correctly if it is converted to all
// lower case.  Converting it to all upper case is allowed.
func Ilaenv2stage(ispec *int, name, opts []byte, n1, n2, n3, n4 *int) (ilaenv2stageReturn int) {
	var iispec int

	switch *ispec {
	case 1:
		goto label10
	case 2:
		goto label10
	case 3:
		goto label10
	case 4:
		goto label10
	case 5:
		goto label10
	}

	//     Invalid value for ISPEC
	ilaenv2stageReturn = -1
	return

label10:
	;

	//     2stage eigenvalues and SVD or related subroutines.
	iispec = 16 + (*ispec)
	ilaenv2stageReturn = Iparam2stage(&iispec, name, opts, n1, n2, n3, n4)
	return
}
