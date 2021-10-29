package matgen

// Dlaran returns a random real number from a uniform (0,1)
// distribution.
func Dlaran(iseed *[]int) (dlaranReturn float64) {
	var one, r float64
	var ipw2, it1, it2, it3, it4, m1, m2, m3, m4 int

	m1 = 494
	m2 = 322
	m3 = 2508
	m4 = 2549
	one = 1.0
	ipw2 = 4096
	r = one / float64(ipw2)

label10:
	;

	//     multiply the seed by the multiplier modulo 2**48
	it4 = (*iseed)[3] * m4
	it3 = it4 / ipw2
	it4 = it4 - ipw2*it3
	it3 = it3 + (*iseed)[2]*m4 + (*iseed)[3]*m3
	it2 = it3 / ipw2
	it3 = it3 - ipw2*it2
	it2 = it2 + (*iseed)[1]*m4 + (*iseed)[2]*m3 + (*iseed)[3]*m2
	it1 = it2 / ipw2
	it2 = it2 - ipw2*it1
	it1 = it1 + (*iseed)[0]*m4 + (*iseed)[1]*m3 + (*iseed)[2]*m2 + (*iseed)[3]*m1
	it1 = it1 % ipw2

	//     return updated seed
	(*iseed)[0] = it1
	(*iseed)[1] = it2
	(*iseed)[2] = it3
	(*iseed)[3] = it4

	//     convert 48-bit integer to a real number in the interval (0,1)
	dlaranReturn = r * (float64(it1) + r*(float64(it2)+r*(float64(it3)+r*float64(it4))))

	if dlaranReturn == 1.0 {
		//        If a real number has n bits of precision, and the first
		//        n bits of the 48-bit integer above happen to be all 1 (which
		//        will occur about once every 2**n calls), then DLARAN will
		//        be rounded to exactly 1.0.
		//        Since DLARAN is not supposed to return exactly 0.0 or 1.0
		//        (and some callers of DLARAN, such as CLARND, depend on that),
		//        the statistically correct thing to do in this situation is
		//        simply to iterate again.
		//        N.B. the case DLARAN = 0.0 should not be possible.
		//
		goto label10
	}

	return
}
