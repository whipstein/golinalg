package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlansb returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the element of  largest absolute value  of an
// n by n symmetric band matrix A,  with k super-diagonals.
func Zlansb(norm, uplo byte, n, k *int, ab *mat.CMatrix, ldab *int, work *mat.Vector) (zlansbReturn float64) {
	var absa, one, sum, value, zero float64
	var i, j, l int
	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find maxint(abs(A(i,j))).
		value = zero
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				for i = maxint((*k)+2-j, 1); i <= (*k)+1; i++ {
					sum = ab.GetMag(i-1, j-1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= minint((*n)+1-j, (*k)+1); i++ {
					sum = ab.GetMag(i-1, j-1)
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
			}
		}
	} else if norm == 'I' || norm == 'O' || norm == '1' {
		//        Find normI(A) ( = norm1(A), since A is symmetric).
		value = zero
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				sum = zero
				l = (*k) + 1 - j
				for i = maxint(1, j-(*k)); i <= j-1; i++ {
					absa = ab.GetMag(l+i-1, j-1)
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
				}
				work.Set(j-1, sum+ab.GetMag((*k)+1-1, j-1))
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
				sum = work.Get(j-1) + ab.GetMag(0, j-1)
				l = 1 - j
				for i = j + 1; i <= minint(*n, j+(*k)); i++ {
					absa = ab.GetMag(l+i-1, j-1)
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
		if (*k) > 0 {
			if uplo == 'U' {
				for j = 2; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint(j-1, *k)), ab.CVector(maxint((*k)+2-j, 1)-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
				l = (*k) + 1
			} else {
				for j = 1; j <= (*n)-1; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint((*n)-j, *k)), ab.CVector(1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
				l = 1
			}
			ssq.Set(1, 2*ssq.Get(1))
		} else {
			l = 1
		}

		//        Sum diagonal
		colssq.Set(0, zero)
		colssq.Set(1, one)
		Zlassq(n, ab.CVector(l-1, 0), ldab, colssq.GetPtr(0), colssq.GetPtr(1))
		Dcombssq(ssq, colssq)
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlansbReturn = value
	return
}
