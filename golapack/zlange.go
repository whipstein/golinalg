package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlange returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// complex matrix A.
func Zlange(norm byte, m, n *int, a *mat.CMatrix, lda *int, work *mat.Vector) (zlangeReturn float64) {
	var one, sum, temp, value, zero float64
	var i, j int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if min(*m, *n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		value = zero
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				temp = a.GetMag(i-1, j-1)
				if value < temp || Disnan(int(temp)) {
					value = temp
				}
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		value = zero
		for j = 1; j <= (*n); j++ {
			sum = zero
			for i = 1; i <= (*m); i++ {
				sum = sum + a.GetMag(i-1, j-1)
			}
			if value < sum || Disnan(int(sum)) {
				value = sum
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		for i = 1; i <= (*m); i++ {
			work.Set(i-1, zero)
		}
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
			}
		}
		value = zero
		for i = 1; i <= (*m); i++ {
			temp = work.Get(i - 1)
			if value < temp || Disnan(int(temp)) {
				value = temp
			}
		}
	} else if norm == 'F' || norm == 'E' {
		//        Find normF(A).
		//        SSQ(1) is scale
		//        SSQ(2) is sum-of-squares
		//        For better accuracy, sum each column separately.
		ssq.Set(0, zero)
		ssq.Set(1, one)
		for j = 1; j <= (*n); j++ {
			colssq.Set(0, zero)
			colssq.Set(1, one)
			Zlassq(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
			Dcombssq(ssq, colssq)
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlangeReturn = value
	return
}
