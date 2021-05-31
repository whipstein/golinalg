package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbtf2 computes an LU factorization of a complex m-by-n band matrix
// A using partial pivoting with row interchanges.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zgbtf2(m, n, kl, ku *int, ab *mat.CMatrix, ldab *int, ipiv *[]int, info *int) {
	var one, zero complex128
	var i, j, jp, ju, km, kv int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     KV is the number of superdiagonals in the factor U, allowing for
	//     fill-in.
	kv = (*ku) + (*kl)

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kl) < 0 {
		(*info) = -3
	} else if (*ku) < 0 {
		(*info) = -4
	} else if (*ldab) < (*kl)+kv+1 {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Gaussian elimination with partial pivoting
	//
	//     Set fill-in elements in columns KU+2 to KV to zero.
	for j = (*ku) + 2; j <= minint(kv, *n); j++ {
		for i = kv - j + 2; i <= (*kl); i++ {
			ab.Set(i-1, j-1, zero)
		}
	}

	//     JU is the index of the last column affected by the current stage
	//     of the factorization.
	ju = 1

	for j = 1; j <= minint(*m, *n); j++ {
		//        Set fill-in elements in column J+KV to zero.
		if j+kv <= (*n) {
			for i = 1; i <= (*kl); i++ {
				ab.Set(i-1, j+kv-1, zero)
			}
		}

		//        Find pivot and test for singularity. KM is the number of
		//        subdiagonal elements in the current column.
		km = minint(*kl, (*m)-j)
		jp = goblas.Izamax(toPtr(km+1), ab.CVector(kv+1-1, j-1), func() *int { y := 1; return &y }())
		(*ipiv)[j-1] = jp + j - 1
		if ab.Get(kv+jp-1, j-1) != zero {
			ju = maxint(ju, minint(j+(*ku)+jp-1, *n))

			//           Apply interchange to columns J to JU.
			if jp != 1 {
				goblas.Zswap(toPtr(ju-j+1), ab.CVector(kv+jp-1, j-1), toPtr((*ldab)-1), ab.CVector(kv+1-1, j-1), toPtr((*ldab)-1))
			}
			if km > 0 {
				//              Compute multipliers.
				goblas.Zscal(&km, toPtrc128(one/ab.Get(kv+1-1, j-1)), ab.CVector(kv+2-1, j-1), func() *int { y := 1; return &y }())

				//              Update trailing submatrix within the band.
				if ju > j {
					goblas.Zgeru(&km, toPtr(ju-j), toPtrc128(-one), ab.CVector(kv+2-1, j-1), func() *int { y := 1; return &y }(), ab.CVector(kv-1, j+1-1), toPtr((*ldab)-1), ab.Off(kv+1-1, j+1-1).UpdateRows((*ldab)-1), toPtr((*ldab)-1))
				}
			}
		} else {
			//           If pivot is zero, set INFO to the index of the pivot
			//           unless a zero pivot has already been found.
			if (*info) == 0 {
				(*info) = j
			}
		}
	}
}
