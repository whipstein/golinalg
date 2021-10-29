package lin

// Icopy copies an integer vector x to an integer vector y.
// Uses unrolled loops for increments equal to 1.
func icopy(n int, sx *[]int, incx int, sy *[]int, incy int) {
	var i, ix, iy, m, mp1 int

	if n <= 0 {
		return
	}
	if incx == 1 && incy == 1 {
		goto label20
	}

	//     Code for unequal increments or equal increments not equal to 1
	ix = 1
	iy = 1
	if incx < 0 {
		ix = (-n+1)*incx + 1
	}
	if incy < 0 {
		iy = (-n+1)*incy + 1
	}
	for i = 1; i <= n; i++ {
		(*sy)[iy-1] = (*sx)[ix-1]
		ix = ix + incx
		iy = iy + incy
	}
	return

	//     Code for both increments equal to 1
	//
	//     Clean-up loop
label20:
	;
	m = n % 7
	if m == 0 {
		goto label40
	}
	for i = 1; i <= m; i++ {
		(*sy)[i-1] = (*sx)[i-1]
	}
	if n < 7 {
		return
	}
label40:
	;
	mp1 = m + 1
	for i = mp1; i <= n; i += 7 {
		(*sy)[i-1] = (*sx)[i-1]
		(*sy)[i] = (*sx)[i]
		(*sy)[i+2-1] = (*sx)[i+2-1]
		(*sy)[i+3-1] = (*sx)[i+3-1]
		(*sy)[i+4-1] = (*sx)[i+4-1]
		(*sy)[i+5-1] = (*sx)[i+5-1]
		(*sy)[i+6-1] = (*sx)[i+6-1]
	}
}
