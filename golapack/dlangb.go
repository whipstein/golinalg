package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlangb returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the element of  largest absolute value  of an
// n by n band matrix  A,  with kl sub-diagonals and ku super-diagonals.
func Dlangb(norm byte, n, kl, ku *int, ab *mat.Matrix, ldab *int, work *mat.Vector) (dlangbReturn float64) {
	var one, sum, temp, value, zero float64
	var i, j, k, l int

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
			for i = maxint((*ku)+2-j, 1); i <= minint((*n)+(*ku)+1-j, (*kl)+(*ku)+1); i++ {
				temp = math.Abs(ab.Get(i-1, j-1))
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
			for i = maxint((*ku)+2-j, 1); i <= minint((*n)+(*ku)+1-j, (*kl)+(*ku)+1); i++ {
				sum = sum + math.Abs(ab.Get(i-1, j-1))
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
			k = (*ku) + 1 - j
			for i = maxint(1, j-(*ku)); i <= minint(*n, j+(*kl)); i++ {
				work.Set(i-1, work.Get(i-1)+math.Abs(ab.Get(k+i-1, j-1)))
			}
		}
		value = zero
		for i = 1; i <= (*n); i++ {
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
			l = maxint(1, j-(*ku))
			k = (*ku) + 1 - j + l
			colssq.Set(0, zero)
			colssq.Set(1, one)
			Dlassq(toPtr(minint(*n, j+(*kl))-l+1), ab.Vector(k-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
			Dcombssq(ssq, colssq)
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	dlangbReturn = value
	return
}
