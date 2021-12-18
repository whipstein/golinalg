package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlantb returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the element of  largest absolute value  of an
// n by n triangular band matrix A,  with ( k + 1 ) diagonals.
func Zlantb(norm byte, uplo mat.MatUplo, diag mat.MatDiag, n, k int, ab *mat.CMatrix, work *mat.Vector) (zlantbReturn float64) {
	var udiag bool
	var one, sum, value, zero float64
	var i, j, l int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if n == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abscmplx.Abs(A(i,j))).
		if diag == Unit {
			value = one
			if uplo == Upper {
				for j = 1; j <= n; j++ {
					for i = max(k+2-j, 1); i <= k; i++ {
						sum = ab.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 2; i <= min(n+1-j, k+1); i++ {
						sum = ab.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			}
		} else {
			value = zero
			if uplo == Upper {
				for j = 1; j <= n; j++ {
					for i = max(k+2-j, 1); i <= k+1; i++ {
						sum = ab.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= min(n+1-j, k+1); i++ {
						sum = ab.GetMag(i-1, j-1)
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
		udiag = diag == Unit
		if uplo == Upper {
			for j = 1; j <= n; j++ {
				if udiag {
					sum = one
					for i = max(k+2-j, 1); i <= k; i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = max(k+2-j, 1); i <= k+1; i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if udiag {
					sum = one
					for i = 2; i <= min(n+1-j, k+1); i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = 1; i <= min(n+1-j, k+1); i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		value = zero
		if uplo == Upper {
			if diag == Unit {
				for i = 1; i <= n; i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= n; j++ {
					l = k + 1 - j
					for i = max(1, j-k); i <= j-1; i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= n; i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= n; j++ {
					l = k + 1 - j
					for i = max(1, j-k); i <= j; i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			}
		} else {
			if diag == Unit {
				for i = 1; i <= n; i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= n; j++ {
					l = 1 - j
					for i = j + 1; i <= min(n, j+k); i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= n; i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= n; j++ {
					l = 1 - j
					for i = j; i <= min(n, j+k); i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			}
		}
		for i = 1; i <= n; i++ {
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
		if uplo == Upper {
			if diag == Unit {
				ssq.Set(0, one)
				ssq.Set(1, float64(n))
				if k > 0 {
					for j = 2; j <= n; j++ {
						colssq.Set(0, zero)
						colssq.Set(1, one)
						*colssq.GetPtr(0), *colssq.GetPtr(1) = Zlassq(min(j-1, k), ab.Off(max(k+2-j, 1)-1, j-1).CVector(), 1, colssq.Get(0), colssq.Get(1))
						Dcombssq(ssq, colssq)
					}
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Zlassq(min(j, k+1), ab.Off(max(k+2-j, 1)-1, j-1).CVector(), 1, colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
				}
			}
		} else {
			if diag == Unit {
				ssq.Set(0, one)
				ssq.Set(1, float64(n))
				if k > 0 {
					for j = 1; j <= n-1; j++ {
						colssq.Set(0, zero)
						colssq.Set(1, one)
						*colssq.GetPtr(0), *colssq.GetPtr(1) = Zlassq(min(n-j, k), ab.Off(1, j-1).CVector(), 1, colssq.Get(0), colssq.Get(1))
						Dcombssq(ssq, colssq)
					}
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Zlassq(min(n-j+1, k+1), ab.Off(0, j-1).CVector(), 1, colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlantbReturn = value
	return
}
