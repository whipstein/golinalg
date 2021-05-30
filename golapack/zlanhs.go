package golapack

import (
	"golinalg/mat"
	"math"
)

// Zlanhs returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// Hessenberg matrix A.
func Zlanhs(norm byte, n *int, a *mat.CMatrix, lda *int, work *mat.Vector) (zlanhsReturn float64) {
	var one, sum, value, zero float64
	var i, j int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		value = zero
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(*n, j+1); i++ {
				sum = a.GetMag(i-1, j-1)
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		value = zero
		for j = 1; j <= (*n); j++ {
			sum = zero
			for i = 1; i <= minint(*n, j+1); i++ {
				sum = sum + a.GetMag(i-1, j-1)
			}
			if value < sum || Disnan(int(sum)) {
				value = sum
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		for i = 1; i <= (*n); i++ {
			work.Set(i-1, zero)
		}
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(*n, j+1); i++ {
				work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
			}
		}
		value = zero
		for i = 1; i <= (*n); i++ {
			sum = work.Get(i - 1)
			if value < sum || Disnan(int(sum)) {
				value = sum
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
			Zlassq(toPtr(minint(*n, j+1)), a.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
			Dcombssq(ssq, colssq)
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlanhsReturn = value
	return
}
