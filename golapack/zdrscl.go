package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zdrscl multiplies an n-element complex vector x by the real scalar
// 1/a.  This is done without overflow or underflow as long as
// the final result x/a does not overflow or underflow.
func Zdrscl(n *int, sa *float64, sx *mat.CVector, incx *int) {
	var done bool
	var bignum, cden, cden1, cnum, cnum1, mul, one, smlnum, zero float64

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Initialize the denominator to SA and the numerator to 1.
	cden = (*sa)
	cnum = one

label10:
	;
	cden1 = cden * smlnum
	cnum1 = cnum / bignum
	if math.Abs(cden1) > math.Abs(cnum) && cnum != zero {
		//        Pre-multiply X by SMLNUM if CDEN is large compared to CNUM.
		mul = smlnum
		done = false
		cden = cden1
	} else if math.Abs(cnum1) > math.Abs(cden) {
		//        Pre-multiply X by BIGNUM if CDEN is small compared to CNUM.
		mul = bignum
		done = false
		cnum = cnum1
	} else {
		//        Multiply X by CNUM / CDEN and return.
		mul = cnum / cden
		done = true
	}

	//     Scale the vector X by MUL
	goblas.Zdscal(*n, mul, sx.Off(0, *incx))

	if !done {
		goto label10
	}
}
