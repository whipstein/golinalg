package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlantr returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// trapezoidal or triangular matrix A.
func Dlantr(norm byte, uplo mat.MatUplo, diag mat.MatDiag, m, n int, a *mat.Matrix, work *mat.Vector) (dlantrReturn float64) {
	var udiag bool
	var one, sum, value, zero float64
	var i, j int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if min(m, n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		if diag == Unit {
			value = one
			if uplo == Upper {
				for j = 1; j <= n; j++ {
					for i = 1; i <= min(m, j-1); i++ {
						sum = math.Abs(a.Get(i-1, j-1))
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j + 1; i <= m; i++ {
						sum = math.Abs(a.Get(i-1, j-1))
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
					for i = 1; i <= min(m, j); i++ {
						sum = math.Abs(a.Get(i-1, j-1))
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= m; i++ {
						sum = math.Abs(a.Get(i-1, j-1))
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
				if udiag && (j <= m) {
					sum = one
					for i = 1; i <= j-1; i++ {
						sum += math.Abs(a.Get(i-1, j-1))
					}
				} else {
					sum = zero
					for i = 1; i <= min(m, j); i++ {
						sum += math.Abs(a.Get(i-1, j-1))
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
					for i = j + 1; i <= m; i++ {
						sum += math.Abs(a.Get(i-1, j-1))
					}
				} else {
					sum = zero
					for i = j; i <= m; i++ {
						sum += math.Abs(a.Get(i-1, j-1))
					}
				}
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		}
	} else if norm == 'I' {
		//        Find normI(A).
		if uplo == Upper {
			if diag == Unit {
				for i = 1; i <= m; i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= n; j++ {
					for i = 1; i <= min(m, j-1); i++ {
						work.Set(i-1, work.Get(i-1)+math.Abs(a.Get(i-1, j-1)))
					}
				}
			} else {
				for i = 1; i <= m; i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= n; j++ {
					for i = 1; i <= min(m, j); i++ {
						work.Set(i-1, work.Get(i-1)+math.Abs(a.Get(i-1, j-1)))
					}
				}
			}
		} else {
			if diag == Unit {
				for i = 1; i <= n; i++ {
					work.Set(i-1, one)
				}
				for i = n + 1; i <= m; i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= n; j++ {
					for i = j + 1; i <= m; i++ {
						work.Set(i-1, work.Get(i-1)+math.Abs(a.Get(i-1, j-1)))
					}
				}
			} else {
				for i = 1; i <= m; i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= n; j++ {
					for i = j; i <= m; i++ {
						work.Set(i-1, work.Get(i-1)+math.Abs(a.Get(i-1, j-1)))
					}
				}
			}
		}
		value = zero
		for i = 1; i <= m; i++ {
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
				ssq.Set(1, float64(min(m, n)))
				for j = 2; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(min(m, j-1), a.Vector(0, j-1, 1), colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
					//Label290:
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(min(m, j), a.Vector(0, j-1, 1), colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
				}
			}
		} else {
			if diag == Unit {
				ssq.Set(0, one)
				ssq.Set(1, float64(min(m, n)))
				for j = 1; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(m-j, a.Vector(min(m, j+1)-1, j-1, 1), colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= n; j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(m-j+1, a.Vector(j-1, j-1, 1), colssq.Get(0), colssq.Get(1))
					Dcombssq(ssq, colssq)
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	dlantrReturn = value
	return
}
