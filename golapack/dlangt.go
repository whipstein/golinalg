package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlangt returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// real tridiagonal matrix A.
func Dlangt(norm byte, n int, dl, d, du *mat.Vector) (dlangtReturn float64) {
	var anorm, one, scale, sum, temp, zero float64
	var i int

	one = 1.0
	zero = 0.0

	if n <= 0 {
		anorm = zero
	} else if norm == 'M' {
		//        Find max(math.Abs(A(i,j))).
		anorm = math.Abs(d.Get(n - 1))
		for i = 1; i <= n-1; i++ {
			if anorm < math.Abs(dl.Get(i-1)) || Disnan(int(math.Abs(dl.Get(i-1)))) {
				anorm = math.Abs(dl.Get(i - 1))
			}
			if anorm < math.Abs(d.Get(i-1)) || Disnan(int(math.Abs(d.Get(i-1)))) {
				anorm = math.Abs(d.Get(i - 1))
			}
			if anorm < math.Abs(du.Get(i-1)) || Disnan(int(math.Abs(du.Get(i-1)))) {
				anorm = math.Abs(du.Get(i - 1))
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		if n == 1 {
			anorm = math.Abs(d.Get(0))
		} else {
			anorm = math.Abs(d.Get(0)) + math.Abs(dl.Get(0))
			temp = math.Abs(d.Get(n-1)) + math.Abs(du.Get(n-1-1))
			if anorm < temp || Disnan(int(temp)) {
				anorm = temp
			}
			for i = 2; i <= n-1; i++ {
				temp = math.Abs(d.Get(i-1)) + math.Abs(dl.Get(i-1)) + math.Abs(du.Get(i-1-1))
				if anorm < temp || Disnan(int(temp)) {
					anorm = temp
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		if n == 1 {
			anorm = math.Abs(d.Get(0))
		} else {
			anorm = math.Abs(d.Get(0)) + math.Abs(du.Get(0))
			temp = math.Abs(d.Get(n-1)) + math.Abs(dl.Get(n-1-1))
			if anorm < temp || Disnan(int(temp)) {
				anorm = temp
			}
			for i = 2; i <= n-1; i++ {
				temp = math.Abs(d.Get(i-1)) + math.Abs(du.Get(i-1)) + math.Abs(dl.Get(i-1-1))
				if anorm < temp || Disnan(int(temp)) {
					anorm = temp
				}
			}
		}
	} else if norm == 'F' || norm == 'E' {
		//        Find normF(A).
		scale = zero
		sum = one
		scale, sum = Dlassq(n, d, 1, scale, sum)
		if n > 1 {
			scale, sum = Dlassq(n-1, dl, 1, scale, sum)
			scale, sum = Dlassq(n-1, du, 1, scale, sum)
		}
		anorm = scale * math.Sqrt(sum)
	}

	dlangtReturn = anorm
	return
}
