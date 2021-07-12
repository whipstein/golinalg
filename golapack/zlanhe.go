package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlanhe returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// complex hermitian matrix A.
func Zlanhe(norm, uplo byte, n *int, a *mat.CMatrix, lda *int, work *mat.Vector) (zlanheReturn float64) {
	var absa, one, sum, value, zero float64
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
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j-1; i++ {
					sum = a.GetMag(i-1, j-1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
				sum = math.Abs(real(a.Get(j-1, j-1)))
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				sum = math.Abs(real(a.Get(j-1, j-1)))
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
				for i = j + 1; i <= (*n); i++ {
					sum = a.GetMag(i-1, j-1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
			}
		}
	} else if norm == 'I' || norm == 'O' || norm == '1' {
		//        Find normI(A) ( = norm1(A), since A is hermitian).
		value = zero
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				sum = zero
				for i = 1; i <= j-1; i++ {
					absa = a.GetMag(i-1, j-1)
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
				}
				work.Set(j-1, sum+math.Abs(real(a.Get(j-1, j-1))))
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
				sum = work.Get(j-1) + math.Abs(real(a.Get(j-1, j-1)))
				for i = j + 1; i <= (*n); i++ {
					absa = a.GetMag(i-1, j-1)
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
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
		if uplo == 'U' {
			for j = 2; j <= (*n); j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				Zlassq(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
			}
		} else {
			for j = 1; j <= (*n)-1; j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				Zlassq(toPtr((*n)-j), a.CVector(j, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
			}
		}
		ssq.Set(1, 2*ssq.Get(1))

		//        Sum diagonal
		for i = 1; i <= (*n); i++ {
			if real(a.Get(i-1, i-1)) != zero {
				absa = math.Abs(real(a.Get(i-1, i-1)))
				if ssq.Get(0) < absa {
					ssq.Set(1, one+ssq.Get(1)*math.Pow(ssq.Get(0)/absa, 2))
					ssq.Set(0, absa)
				} else {
					ssq.Set(1, ssq.Get(1)+math.Pow(absa/ssq.Get(0), 2))
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlanheReturn = value
	return
}
