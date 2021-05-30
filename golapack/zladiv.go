package golapack

// Zladiv := X / Y, where X and Y are complex.  The computation of X / Y
// will not overflow on an intermediary step unless the results
// overflows.
func Zladiv(x, y *complex128) (zladivReturn complex128) {
	var zi, zr float64

	Dladiv(toPtrf64(real(*x)), toPtrf64(imag(*x)), toPtrf64(real(*y)), toPtrf64(imag(*y)), &zr, &zi)
	zladivReturn = complex(zr, zi)

	return
}
