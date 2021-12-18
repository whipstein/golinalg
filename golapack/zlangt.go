package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlangt returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// complex tridiagonal matrix A.
func Zlangt(norm byte, n int, dl, d, du *mat.CVector) (zlangtReturn float64) {
	var anorm, one, scale, sum, temp, zero float64
	var i int

	one = 1.0
	zero = 0.0

	if n <= 0 {
		anorm = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		anorm = d.GetMag(n - 1)
		for i = 1; i <= n-1; i++ {
			if anorm < dl.GetMag(i-1) || Disnan(int(dl.GetMag(i-1))) {
				anorm = dl.GetMag(i - 1)
			}
			if anorm < d.GetMag(i-1) || Disnan(int(d.GetMag(i-1))) {
				anorm = d.GetMag(i - 1)
			}
			if anorm < du.GetMag(i-1) || Disnan(int(du.GetMag(i-1))) {
				anorm = du.GetMag(i - 1)
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		if n == 1 {
			anorm = d.GetMag(0)
		} else {
			anorm = d.GetMag(0) + dl.GetMag(0)
			temp = d.GetMag(n-1) + du.GetMag(n-1-1)
			if anorm < temp || Disnan(int(temp)) {
				anorm = temp
			}
			for i = 2; i <= n-1; i++ {
				temp = d.GetMag(i-1) + dl.GetMag(i-1) + du.GetMag(i-1-1)
				if anorm < temp || Disnan(int(temp)) {
					anorm = temp
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		if n == 1 {
			anorm = d.GetMag(0)
		} else {
			anorm = d.GetMag(0) + du.GetMag(0)
			temp = d.GetMag(n-1) + dl.GetMag(n-1-1)
			if anorm < temp || Disnan(int(temp)) {
				anorm = temp
			}
			for i = 2; i <= n-1; i++ {
				temp = d.GetMag(i-1) + du.GetMag(i-1) + dl.GetMag(i-1-1)
				if anorm < temp || Disnan(int(temp)) {
					anorm = temp
				}
			}
		}
	} else if norm == 'F' || norm == 'E' {
		//        Find normF(A).
		scale = zero
		sum = one
		scale, sum = Zlassq(n, d, 1, scale, sum)
		if n > 1 {
			scale, sum = Zlassq(n-1, dl, 1, scale, sum)
			scale, sum = Zlassq(n-1, du, 1, scale, sum)
		}
		anorm = scale * math.Sqrt(sum)
	}

	zlangtReturn = anorm
	return
}
