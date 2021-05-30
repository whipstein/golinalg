package golapack

import (
	"golinalg/mat"
	"math"
)

// Zlantr returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// trapezoidal or triangular matrix A.
func Zlantr(norm, uplo, diag byte, m, n *int, a *mat.CMatrix, lda *int, work *mat.Vector) (zlantrReturn float64) {
	var udiag bool
	var one, sum, value, zero float64
	var i, j int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if minint(*m, *n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		if diag == 'U' {
			value = one
			if uplo == 'U' {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= minint(*m, j-1); i++ {
						sum = a.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= (*n); j++ {
					for i = j + 1; i <= (*m); i++ {
						sum = a.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			}
		} else {
			value = zero
			if uplo == 'U' {
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= minint(*m, j); i++ {
						sum = a.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= (*n); j++ {
					for i = j; i <= (*m); i++ {
						sum = a.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		value = zero
		udiag = diag == 'U'
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				if udiag && (j <= (*m)) {
					sum = one
					for i = 1; i <= j-1; i++ {
						sum = sum + a.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = 1; i <= minint(*m, j); i++ {
						sum = sum + a.GetMag(i-1, j-1)
					}
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if udiag {
					sum = one
					for i = j + 1; i <= (*m); i++ {
						sum = sum + a.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = j; i <= (*m); i++ {
						sum = sum + a.GetMag(i-1, j-1)
					}
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		if uplo == 'U' {
			if diag == 'U' {
				for i = 1; i <= (*m); i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= minint(*m, j-1); i++ {
						work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= (*m); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= minint(*m, j); i++ {
						work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
					}
				}
			}
		} else {
			if diag == 'U' {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, one)
				}
				for i = (*n) + 1; i <= (*m); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					for i = j + 1; i <= (*m); i++ {
						work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= (*m); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					for i = j; i <= (*m); i++ {
						work.Set(i-1, work.Get(i-1)+a.GetMag(i-1, j-1))
					}
				}
			}
		}
		value = zero
		for i = 1; i <= (*m); i++ {
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
		if uplo == 'U' {
			if diag == 'U' {
				ssq.Set(0, one)
				ssq.Set(1, float64(minint(*m, *n)))
				for j = 2; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint(*m, j-1)), a.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint(*m, j)), a.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			}
		} else {
			if diag == 'U' {
				ssq.Set(0, one)
				ssq.Set(1, float64(minint(*m, *n)))
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr((*m)-j), a.CVector(minint(*m, j+1)-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr((*m)-j+1), a.CVector(j-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlantrReturn = value
	return
}
