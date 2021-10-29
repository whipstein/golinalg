package golapack

// Ieeeck is called from the ILAENV to verify that Infinity and
// possibly NaN arithmetic is safe (i.e. will not trap).
func Ieeeck(ispec int, zero, one float64) (ieeeckReturn int) {
	var nan1, nan2, nan4, nan5, nan6, neginf, negzro, newzro, posinf float64

	ieeeckReturn = 1

	posinf = one / zero
	if posinf <= one {
		ieeeckReturn = 0
		return
	}

	neginf = -one / zero
	if neginf >= zero {
		ieeeckReturn = 0
		return
	}

	negzro = one / (neginf + one)
	if negzro != zero {
		ieeeckReturn = 0
		return
	}

	neginf = one / negzro
	if neginf >= zero {
		ieeeckReturn = 0
		return
	}

	newzro = negzro + zero
	if newzro != zero {
		ieeeckReturn = 0
		return
	}

	posinf = one / newzro
	if posinf <= one {
		ieeeckReturn = 0
		return
	}

	neginf = neginf * posinf
	if neginf >= zero {
		ieeeckReturn = 0
		return
	}

	posinf = posinf * posinf
	if posinf <= one {
		ieeeckReturn = 0
		return
	}

	//     Return if we were only asked to check infinity arithmetic
	if ispec == 0 {
		return
	}

	nan1 = posinf + neginf

	nan2 = posinf / neginf

	// nan3 = posinf / posinf

	nan4 = posinf * zero

	nan5 = neginf * negzro

	nan6 = nan5 * zero

	if nan1 == nan1 {
		ieeeckReturn = 0
		return
	}

	if nan2 == nan2 {
		ieeeckReturn = 0
		return
	}

	// if nan3 == nan3 {
	// 	ieeeckReturn = 0
	// 	return
	// }

	if nan4 == nan4 {
		ieeeckReturn = 0
		return
	}

	if nan5 == nan5 {
		ieeeckReturn = 0
		return
	}

	if nan6 == nan6 {
		ieeeckReturn = 0
		return
	}

	return
}
