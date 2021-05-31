package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlantb returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the element of  largest absolute value  of an
// n by n triangular band matrix A,  with ( k + 1 ) diagonals.
func Zlantb(norm, uplo, diag byte, n, k *int, ab *mat.CMatrix, ldab *int, work *mat.Vector) (zlantbReturn float64) {
	var udiag bool
	var one, sum, value, zero float64
	var i, j, l int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if (*n) == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find maxint(abscmplx.Abs(A(i,j))).
		if diag == 'U' {
			value = one
			if uplo == 'U' {
				for j = 1; j <= (*n); j++ {
					for i = maxint((*k)+2-j, 1); i <= (*k); i++ {
						sum = ab.GetMag(i-1, j-1)
						if value < sum || Disnan(int(sum)) {
							value = sum
						}
					}
				}
			} else {
				for j = 1; j <= (*n); j++ {
					for i = 2; i <= minint((*n)+1-j, (*k)+1); i++ {
						sum = ab.GetMag(i-1, j-1)
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
		}
	} else if norm == 'O' || norm == '1' {
		//        Find norm1(A).
		value = zero
		udiag = diag == 'U'
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				if udiag {
					sum = one
					for i = maxint((*k)+2-j, 1); i <= (*k); i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = maxint((*k)+2-j, 1); i <= (*k)+1; i++ {
						sum = sum + ab.GetMag(i-1, j-1)
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
					for i = 2; i <= minint((*n)+1-j, (*k)+1); i++ {
						sum = sum + ab.GetMag(i-1, j-1)
					}
				} else {
					sum = zero
					for i = 1; i <= minint((*n)+1-j, (*k)+1); i++ {
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
		if uplo == 'U' {
			if diag == 'U' {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= (*n); j++ {
					l = (*k) + 1 - j
					for i = maxint(1, j-(*k)); i <= j-1; i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					l = (*k) + 1 - j
					for i = maxint(1, j-(*k)); i <= j; i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			}
		} else {
			if diag == 'U' {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, one)
				}
				for j = 1; j <= (*n); j++ {
					l = 1 - j
					for i = j + 1; i <= minint(*n, j+(*k)); i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			} else {
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, zero)
				}
				for j = 1; j <= (*n); j++ {
					l = 1 - j
					for i = j; i <= minint(*n, j+(*k)); i++ {
						work.Set(i-1, work.Get(i-1)+ab.GetMag(l+i-1, j-1))
					}
				}
			}
		}
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
				if (*k) > 0 {
					for j = 2; j <= (*n); j++ {
						colssq.Set(0, zero)
						colssq.Set(1, one)
						Zlassq(toPtr(minint(j-1, *k)), ab.CVector(maxint((*k)+2-j, 1)-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
						Dcombssq(ssq, colssq)
					}
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint(j, (*k)+1)), ab.CVector(maxint((*k)+2-j, 1)-1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			}
		} else {
			if diag == 'U' {
				ssq.Set(0, one)
				ssq.Set(1, float64(*n))
				if (*k) > 0 {
					for j = 1; j <= (*n)-1; j++ {
						colssq.Set(0, zero)
						colssq.Set(1, one)
						Zlassq(toPtr(minint((*n)-j, *k)), ab.CVector(1, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
						Dcombssq(ssq, colssq)
					}
				}
			} else {
				ssq.Set(0, zero)
				ssq.Set(1, one)
				for j = 1; j <= (*n); j++ {
					colssq.Set(0, zero)
					colssq.Set(1, one)
					Zlassq(toPtr(minint((*n)-j+1, (*k)+1)), ab.CVector(0, j-1), func() *int { y := 1; return &y }(), colssq.GetPtr(0), colssq.GetPtr(1))
					Dcombssq(ssq, colssq)
				}
			}
		}
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	zlantbReturn = value
	return
}
