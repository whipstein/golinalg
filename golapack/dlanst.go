package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlanst returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// real symmetric tridiagonal matrix A.
// \endverbatim
//
// \return DLANST
// \verbatim
//
//    DLANST = ( max(math.Abs(A(i,j))), NORM = 'M' or 'm'
//             (
//             ( norm1(A),         NORM = '1', 'O' or 'o'
//             (
//             ( normI(A),         NORM = 'I' or 'i'
//             (
//             ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
//
// where  norm1  denotes the  one norm of a matrix (maximum column sum),
// normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
// normF  denotes the  Frobenius norm of a matrix (square root of sum of
// squares).  Note that  max(math.Abs(A(i,j)))  is not a consistent matrix norm.
func Dlanst(norm byte, n *int, d, e *mat.Vector) (dlanstReturn float64) {
	var anorm, one, scale, sum, zero float64
	var i int

	one = 1.0
	zero = 0.0

	if (*n) <= 0 {
		anorm = zero
	} else if norm == 'M' {
		//        Find max(math.Abs(A(i,j))).
		anorm = math.Abs(d.Get((*n) - 1))
		for i = 1; i <= (*n)-1; i++ {
			sum = math.Abs(d.Get(i - 1))
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
			sum = math.Abs(e.Get(i - 1))
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
		}
	} else if norm == 'O' || norm == '1' || norm == 'I' {
		//        Find norm1(A).
		if (*n) == 1 {
			anorm = math.Abs(d.Get(0))
		} else {
			anorm = math.Abs(d.Get(0)) + math.Abs(e.Get(0))
			sum = math.Abs(e.Get((*n)-1-1)) + math.Abs(d.Get((*n)-1))
			if anorm < sum || Disnan(int(sum)) {
				anorm = sum
			}
			for i = 2; i <= (*n)-1; i++ {
				sum = math.Abs(d.Get(i-1)) + math.Abs(e.Get(i-1)) + math.Abs(e.Get(i-1-1))
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
			Dlassq(toPtr((*n)-1), e, func() *int { y := 1; return &y }(), &scale, &sum)
			sum = 2 * sum
		}
		Dlassq(n, d, func() *int { y := 1; return &y }(), &scale, &sum)
		anorm = scale * math.Sqrt(sum)
	}

	dlanstReturn = anorm
	return
}
