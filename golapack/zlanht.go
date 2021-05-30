package golapack

import (
	"golinalg/mat"
	"math"
)

// Zlanht returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// complex Hermitian tridiagonal matrix A.
func Zlanht(norm byte, n *int, d *mat.Vector, e *mat.CVector) (zlanhtReturn float64) {
	var anorm, one, scale, sum, zero float64
	var i int

	one = 1.0
	zero = 0.0

	if (*n) <= 0 {
		anorm = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		anorm = math.Abs(d.Get((*n) - 1))
		for i = 1; i <= (*n)-1; i++ {
			sum = math.Abs(d.Get(i - 1))
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
			sum = e.GetMag(i - 1)
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
		}
	} else if norm == 'O' || norm == '1' || norm == 'I' {
		//        Find norm1(A).
		if (*n) == 1 {
			anorm = math.Abs(d.Get(0))
		} else {
			anorm = math.Abs(d.Get(0)) + e.GetMag(0)
			sum = e.GetMag((*n)-1-1) + math.Abs(d.Get((*n)-1))
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
			for i = 2; i <= (*n)-1; i++ {
				sum = math.Abs(d.Get(i-1)) + e.GetMag(i-1) + e.GetMag(i-1-1)
				if anorm < sum || Disnan(int(sum)) {
					anorm = sum
				}
			}
		}
	} else if norm == 'F' || norm == 'E' {
		//        Find normF(A).
		scale = zero
		sum = one
		if (*n) > 1 {
			Zlassq(toPtr((*n)-1), e, func() *int { y := 1; return &y }(), &scale, &sum)
			sum = 2 * sum
		}
		Dlassq(n, d, func() *int { y := 1; return &y }(), &scale, &sum)
		anorm = scale * math.Sqrt(sum)
	}

	zlanhtReturn = anorm
	return
}
