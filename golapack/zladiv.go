package golapack

// Zladiv := X / Y, where X and Y are complex.  The computation of X / Y
// will not overflow on an intermediary step unless the results
// overflows.
func Zladiv(x, y complex128) (zladivReturn complex128) {
	var zi, zr float64

	zr, zi = Dladiv(real(x), imag(x), real(y), imag(y))
	zladivReturn = complex(zr, zi)

	return
}
