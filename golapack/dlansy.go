package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlansy returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// real symmetric matrix A.
func Dlansy(norm, uplo byte, n *int, a *mat.Matrix, lda *int, work *mat.Vector) (dlansyReturn float64) {
	var absa, one, sum, value, zero float64
	var i, j int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(math.Abs(A(i,j))).
		value = zero
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= j; i++ {
					sum = math.Abs(a.Get(i-1, j-1))
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				for i = j; i <= (*n); i++ {
					sum = math.Abs(a.Get(i-1, j-1))
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
				for i = 1; i <= j-1; i++ {
					absa = math.Abs(a.Get(i-1, j-1))
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
				}
				work.Set(j-1, sum+math.Abs(a.Get(j-1, j-1)))
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
				sum = work.Get(j-1) + math.Abs(a.Get(j-1, j-1))
				for i = j + 1; i <= (*n); i++ {
					absa = math.Abs(a.Get(i-1, j-1))
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
				Dlassq(toPtr(j-1), a.Vector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
			}
		} else {
			for j = 1; j <= (*n)-1; j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				Dlassq(toPtr((*n)-j), a.Vector(j+1-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
				Dcombssq(ssq, colssq)
			}
		}
		ssq.Set(1, 2*ssq.Get(1))

		//        Sum diagonal
		colssq.Set(0, zero)
		colssq.Set(1, one)
		Dlassq(n, a.VectorIdx(0), toPtr((*lda)+1), colssq.GetPtr(0), colssq.GetPtr(1))
		Dcombssq(ssq, colssq)
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	dlansyReturn = value
	return
}
