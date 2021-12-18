package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbtf2 computes an LU factorization of a complex m-by-n band matrix
// A using partial pivoting with row interchanges.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zgbtf2(m, n, kl, ku int, ab *mat.CMatrix, ipiv *[]int) (info int, err error) {
	var one, zero complex128
	var i, j, jp, ju, km, kv int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

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
		err = fmt.Errorf("ab.Rows < kl+kv+1: ab.Rows=%v, kl=%v, kv=%v", ab.Rows, kl, kv)
	}
	if err != nil {
		gltest.Xerbla2("Zgbtf2", err)
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
		jp = ab.Off(kv, j-1).CVector().Iamax(km+1, 1)
		(*ipiv)[j-1] = jp + j - 1
		if ab.Get(kv+jp-1, j-1) != zero {
			ju = max(ju, min(j+ku+jp-1, n))

			//           Apply interchange to columns J to JU.
			if jp != 1 {
				ab.Off(kv, j-1).CVector().Swap(ju-j+1, ab.Off(kv+jp-1, j-1).CVector(), ab.Rows-1, ab.Rows-1)
			}
			if km > 0 {
				//              Compute multipliers.
				ab.Off(kv+2-1, j-1).CVector().Scal(km, one/ab.Get(kv, j-1), 1)

				//              Update trailing submatrix within the band.
				if ju > j {
					if err = ab.Off(kv, j).UpdateRows(ab.Rows-1).Geru(km, ju-j, -one, ab.Off(kv+2-1, j-1).CVector(), 1, ab.Off(kv-1, j).CVector(), ab.Rows-1); err != nil {
						panic(err)
					}
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
