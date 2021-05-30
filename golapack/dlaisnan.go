package golapack

// Dlaisnan routine is not for general use.  It exists solely to avoid
// over-optimization in DISNAN.
//
// DLAISNAN checks for NaNs by comparing its two arguments for
// inequality.  NaN is the only floating-point value where NaN != NaN
// returns .TRUE.  To check for NaNs, pass the same variable as both
// arguments.
//
// A compiler must assume that the two arguments are
// not the same variable, and the test will not be optimized away.
// Interprocedural or whole-program optimization may delete this
// test.  The ISNAN functions will be replaced by the correct
// Fortran 03 intrinsic once the intrinsic is widely available.
func Dlaisnan(din1 int, din2 float64) (dlaisnanReturn bool) {
	dlaisnanReturn = (float64(din1) != din2) || (din1 != int(din2))
	return
}
