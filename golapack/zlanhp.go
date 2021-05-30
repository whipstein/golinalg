package golapack

import (
	"golinalg/mat"
	"math"
)

// Zlanhp returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// complex hermitian matrix A,  supplied in packed form.
func Zlanhp(norm, uplo byte, n *int, ap *mat.CVector, work *mat.Vector) (zlanhpReturn float64) {
	var absa, one, sum, value, zero float64
	var i, j, k int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		value = zero
		if uplo == 'U' {
			k = 0
			for j = 1; j <= (*n); j++ {
				for i = k + 1; i <= k+j-1; i++ {
					sum = ap.GetMag(i - 1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
				k = k + j
				sum = math.Abs(ap.GetRe(k - 1))
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			k = 1
			for j = 1; j <= (*n); j++ {
				sum = math.Abs(ap.GetRe(k - 1))
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
				for i = k + 1; i <= k+(*n)-j; i++ {
					sum = ap.GetMag(i - 1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
				k = k + (*n) - j + 1
			}
		}
	} else if norm == 'I' || norm == 'O' || norm == '1' {
		//        Find normI(A) ( = norm1(A), since A is hermitian).
		value = zero
		k = 1
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				sum = zero
				for i = 1; i <= j-1; i++ {
					absa = ap.GetMag(k - 1)
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
					k = k + 1
				}
				work.Set(j-1, sum+math.Abs(ap.GetRe(k-1)))
				k = k + 1
			}
			for i = 1; i <= (*n); i++ {
				sum = work.Get(i - 1)
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for i = 1; i <= (*n); i++ {
				work.Set(i-1, zero)
			}
			for j = 1; j <= (*n); j++ {
				sum = work.Get(j-1) + math.Abs(ap.GetRe(k-1))
				k = k + 1
				for i = j + 1; i <= (*n); i++ {
					absa = ap.GetMag(k - 1)
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
					k = k + 1
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'F' || norm == 'E' {
		//        Find normF(A).
		//        SSQ(1) is scale
		//        SSQ(2) is sum-of-squares
		//        For better accuracy, sum each column separately.
		ssq.Set(0, zero)
		ssq.Set(1, one)

		//        Sum off-diagonals
		k = 2
		if uplo == 'U' {
			for j = 2; j <= (*n); j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				Zlassq(toPtr(j-1), ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
				k = k + j
			}
		} else {
			for j = 1; j <= (*n)-1; j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				Zlassq(toPtr((*n)-j), ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
				k = k + (*n) - j + 1
			}
		}
		ssq.Set(1, 2*ssq.Get(1))

		//        Sum diagonal
		k = 1
		colssq.Set(0, zero)
		colssq.Set(1, one)
		for i = 1; i <= (*n); i++ {
			if ap.GetRe(k-1) != zero {
				absa = math.Abs(ap.GetRe(k - 1))
				if colssq.Get(0) < absa {
					colssq.Set(1, one+colssq.Get(1)*math.Pow(colssq.Get(0)/absa, 2))
					colssq.Set(0, absa)
				} else {
					colssq.Set(1, colssq.Get(1)+math.Pow(absa/colssq.Get(0), 2))
				}
			}
			if uplo == 'U' {
				k = k + i + 1
			} else {
				k = k + (*n) - i + 1
			}
		}
		Dcombssq(ssq, colssq)
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlanhpReturn = value
	return
}
