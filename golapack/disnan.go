package golapack

// Disnan returns .TRUE. if its argument is NaN, and .FALSE.
// otherwise.  To be replaced by the Fortran 2003 intrinsic in the
// future.
func Disnan(din int) (disnanReturn bool) {
	disnanReturn = Dlaisnan(din, float64(din))
	return
}
