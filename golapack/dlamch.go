package golapack

type MachParam int

const (
	Epsilon MachParam = iota
	Overflow
	Underflow
	SafeMinimum
	Precision
	Base
	Digits
	Round
	MinExponent
	MaxExponent
)

// Dlamch determines double precision machine parameters.
func Dlamch(cmach MachParam) (dlamchReturn float64) {
	var eps, one, rnd, sfmin, small, zero float64

	one = 1.0
	zero = 0.0

	//     Assume rounding, not chopping. Always.
	rnd = one

	if one == rnd {
		eps = epsf64 * 0.5
	} else {
		eps = epsf64
	}

	if cmach == Epsilon {
		dlamchReturn = eps
	} else if cmach == SafeMinimum {
		sfmin = tiny
		small = one / huge
		if small >= sfmin {
			//
			//           Use SMALL plus a bit, to avoid the possibility of rounding
			//           causing overflow when computing  1/sfmin.
			//
			sfmin = small * (one + eps)
		}
		dlamchReturn = sfmin
	} else if cmach == Base {
		dlamchReturn = radix
	} else if cmach == Precision {
		dlamchReturn = eps * radix
	} else if cmach == Digits {
		dlamchReturn = digits
	} else if cmach == Round {
		dlamchReturn = rnd
	} else if cmach == MinExponent {
		dlamchReturn = minexp
	} else if cmach == Underflow {
		dlamchReturn = tiny
	} else if cmach == MaxExponent {
		dlamchReturn = maxexp
	} else if cmach == Overflow {
		dlamchReturn = huge
	} else {
		dlamchReturn = zero
	}

	return
}

// Dlamc3 is intended to force  A  and  B  to be stored prior to doing
// the addition of  A  and  B ,  for use in situations where optimizers
// might hold one of these in a register.
func Dlamc3(a, b *float64) (dlamc3Return float64) {
	return (*a) + (*b)
}
