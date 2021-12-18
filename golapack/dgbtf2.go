package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgbtf2 computes an LU factorization of a real m-by-n band matrix A
// using partial pivoting with row interchanges.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Dgbtf2(m, n, kl, ku int, ab *mat.Matrix, ipiv *[]int) (info int, err error) {
	var one, zero float64
	var i, j, jp, ju, km, kv int

	one = 1.0
	zero = 0.0

	//     KV is the number of superdiagonals in the factor U, allowing for
	//     fill-in.
	kv = ku + kl

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if ab.Rows < kl+kv+1 {
		err = fmt.Errorf("ab.Rows < kl+ku+kl+1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	}
	if err != nil {
		gltest.Xerbla2("Dgbtf2", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Gaussian elimination with partial pivoting
	//
	//     Set fill-in elements in columns KU+2 to KV to zero.
	for j = ku + 2; j <= min(kv, n); j++ {
		for i = kv - j + 2; i <= kl; i++ {
			ab.Set(i-1, j-1, zero)
		}
	}

	//     JU is the index of the last column affected by the current stage
	//     of the factorization.
	ju = 1

	for j = 1; j <= min(m, n); j++ {
		//        Set fill-in elements in column J+KV to zero.
		if j+kv <= n {
			for i = 1; i <= kl; i++ {
				ab.Set(i-1, j+kv-1, zero)
			}
		}

		//        Find pivot and test for singularity. KM is the number of
		//        subdiagonal elements in the current column.
		km = min(kl, m-j)
		jp = ab.Off(kv, j-1).Vector().Iamax(km+1, 1)
		(*ipiv)[j-1] = jp + j - 1
		if ab.Get(kv+jp-1, j-1) != zero {
			ju = max(ju, min(j+ku+jp-1, n))

			//           Apply interchange to columns J to JU.
			if jp != 1 {
				ab.Off(kv, j-1).Vector().Swap(ju-j+1, ab.Off(kv+jp-1, j-1).Vector(), ab.Rows-1, ab.Rows-1)
			}

			if km > 0 {
				//              Compute multipliers.
				ab.Off(kv+2-1, j-1).Vector().Scal(km, one/ab.Get(kv, j-1), 1)

				//              Update trailing submatrix within the band.
				if ju > j {
					err = ab.Off(kv, j).UpdateRows(ab.Rows-1).Ger(km, ju-j, -one, ab.Off(kv+2-1, j-1).Vector(), 1, ab.Off(kv-1, j).Vector(), ab.Rows-1)
				}
			}
		} else {
			//           If pivot is zero, set INFO to the index of the pivot
			//           unless a zero pivot has already been found.
			if info == 0 {
				info = j
			}
		}
	}

	return
}
