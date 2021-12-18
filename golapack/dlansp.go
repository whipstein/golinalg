package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlansp returns the value of the one norm,  or the Frobenius norm, or
// the  infinity norm,  or the  element of  largest absolute value  of a
// real symmetric matrix A,  supplied in packed form.
// \endverbatim
//
// \return DLANSP
// \verbatim
//
//    DLANSP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
//             (
//             ( norm1(A),         NORM = '1', 'O' or 'o'
//             (
//             ( normI(A),         NORM = 'I' or 'i'
//             (
//             ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
//
// where  norm1  denotes the  one norm of a matrix (maximum column sum),
// normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
// normF  denotes the  Frobenius norm of a matrix (square root of sum of
// squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.
func Dlansp(norm byte, uplo mat.MatUplo, n int, ap, work *mat.Vector) (dlanspReturn float64) {
	var absa, one, sum, value, zero float64
	var i, j, k int

	colssq := vf(2)
	ssq := vf(2)

	one = 1.0
	zero = 0.0

	if n == 0 {
		value = zero
	} else if norm == 'M' {
		//        Find max(abs(A(i,j))).
		value = zero
		if uplo == Upper {
			k = 1
			for j = 1; j <= n; j++ {
				for i = k; i <= k+j-1; i++ {
					sum = math.Abs(ap.Get(i - 1))
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
				k = k + j
			}
		} else {
			k = 1
			for j = 1; j <= n; j++ {
				for i = k; i <= k+n-j; i++ {
					sum = math.Abs(ap.Get(i - 1))
					if value < sum || Disnan(int(sum)) {
						value = sum
					}
				}
				k = k + n - j + 1
			}
		}
	} else if norm == 'I' || norm == 'O' || norm == '1' {
		//        Find normI(A) ( = norm1(A), since A is symmetric).
		value = zero
		k = 1
		if uplo == Upper {
			for j = 1; j <= n; j++ {
				sum = zero
				for i = 1; i <= j-1; i++ {
					absa = math.Abs(ap.Get(k - 1))
					sum = sum + absa
					work.Set(i-1, work.Get(i-1)+absa)
					k = k + 1
				}
				work.Set(j-1, sum+math.Abs(ap.Get(k-1)))
				k = k + 1
			}
			for i = 1; i <= n; i++ {
				sum = work.Get(i - 1)
				if value < sum || Disnan(int(sum)) {
					value = sum
				}
			}
		} else {
			for i = 1; i <= n; i++ {
				work.Set(i-1, zero)
			}
			for j = 1; j <= n; j++ {
				sum = work.Get(j-1) + math.Abs(ap.Get(k-1))
				k = k + 1
				for i = j + 1; i <= n; i++ {
					absa = math.Abs(ap.Get(k - 1))
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
		if uplo == Upper {
			for j = 2; j <= n; j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(j-1, ap.Off(k-1), 1, colssq.Get(0), colssq.Get(1))
				Dcombssq(ssq, colssq)
				k = k + j
			}
		} else {
			for j = 1; j <= n-1; j++ {
				colssq.Set(0, zero)
				colssq.Set(1, one)
				*colssq.GetPtr(0), *colssq.GetPtr(1) = Dlassq(n-j, ap.Off(k-1), 1, colssq.Get(0), colssq.Get(1))
				Dcombssq(ssq, colssq)
				k = k + n - j + 1
			}
		}
		ssq.Set(1, 2*ssq.Get(1))

		//        Sum diagonal
		k = 1
		colssq.Set(0, zero)
		colssq.Set(1, one)
		for i = 1; i <= n; i++ {
			if ap.Get(k-1) != zero {
				absa = math.Abs(ap.Get(k - 1))
				if colssq.Get(0) < absa {
					colssq.Set(1, one+colssq.Get(1)*math.Pow(colssq.Get(0)/absa, 2))
					colssq.Set(0, absa)
				} else {
					colssq.Set(1, colssq.Get(1)+math.Pow(absa/colssq.Get(0), 2))
				}
			}
			if uplo == Upper {
				k = k + i + 1
			} else {
				k = k + n - i + 1
			}
		}
		Dcombssq(ssq, colssq)
		value = ssq.Get(0) * math.Sqrt(ssq.Get(1))
	}

	dlanspReturn = value
	return
}
