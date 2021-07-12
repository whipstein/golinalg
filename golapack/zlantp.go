package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlantp returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// triangular matrix A, supplied in packed form.
func Zlantp(norm, uplo, diag byte, n *int, ap *mat.CVector, work *mat.Vector) (zlantpReturn float64) {
	var udiag bool
	var one, sum, value, zero float64
	var i, j, k int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		k = 1
		if diag == 'U' {
			value = one
			if uplo == 'U' {
				for j = 1; j <= (*n); j++ {
					for i = k; i <= k+j-2; i++ {
						sum = ap.GetMag(i - 1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
					k = k + j
				}
			} else {
				for j = 1; j <= (*n); j++ {
					for i = k + 1; i <= k+(*n)-j; i++ {
						sum = ap.GetMag(i - 1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
					k = k + (*n) - j + 1
				}
			}
		} else {
			value = zero
			if uplo == 'U' {
				for j = 1; j <= (*n); j++ {
					for i = k; i <= k+j-1; i++ {
						sum = ap.GetMag(i - 1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
					k = k + j
				}
			} else {
				for j = 1; j <= (*n); j++ {
					for i = k; i <= k+(*n)-j; i++ {
						sum = ap.GetMag(i - 1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
					k = k + (*n) - j + 1
				}
			}
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		value = zero
		k = 1
		udiag = diag == 'U'
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				if udiag {
					sum = one
					for i = k; i <= k+j-2; i++ {
						sum = sum + ap.GetMag(i-1)
					}
				} else {
					sum = zero
					for i = k; i <= k+j-1; i++ {
						sum = sum + ap.GetMag(i-1)
					}
				}
				k = k + j
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				if udiag {
					sum = one
					for i = k + 1; i <= k+(*n)-j; i++ {
						sum = sum + ap.GetMag(i-1)
					}
				} else {
					sum = zero
					for i = k; i <= k+(*n)-j; i++ {
						sum = sum + ap.GetMag(i-1)
					}
				}
				k = k + (*n) - j + 1
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		k = 1
		if uplo == 'U' {
			if diag == 'U' {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= j-1; i++ {
						work.Set(i-1, work.Get(i-1)+ap.GetMag(k-1))
						k = k + 1
					}
					k = k + 1
				}
			} else {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					for i = 1; i <= j; i++ {
						work.Set(i-1, work.Get(i-1)+ap.GetMag(k-1))
						k = k + 1
					}
				}
			}
		} else {
			if diag == 'U' {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= (*n); j++ {
					k = k + 1
					for i = j + 1; i <= (*n); i++ {
						work.Set(i-1, work.Get(i-1)+ap.GetMag(k-1))
						k = k + 1
					}
				}
			} else {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					for i = j; i <= (*n); i++ {
						work.Set(i-1, work.Get(i-1)+ap.GetMag(k-1))
						k = k + 1
					}
				}
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
		if uplo == 'U' {
			if diag == 'U' {
				ssq.Set(0, one)
				ssq.Set(1, float64(*n))
				k = 2
				for j = 2; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(j-1), ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
					k = k + j
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				k = 1
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(&j, ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
					k = k + j
				}
			}
		} else {
			if diag == 'U' {
				ssq.Set(0, one)
				ssq.Set(1, float64(*n))
				k = 2
				for j = 1; j <= (*n)-1; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr((*n)-j), ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
					k = k + (*n) - j + 1
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				k = 1
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr((*n)-j+1), ap.Off(k-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
					k = k + (*n) - j + 1
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlantpReturn = value
	return
}
