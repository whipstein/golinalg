package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbstf computes a split Cholesky factorization of a real
// symmetric positive definite band matrix A.
//
// This routine is designed to be used in conjunction with DSBGST.
//
// The factorization has the form  A = S**T*S  where S is a band matrix
// of the same bandwidth as A and the following structure:
//
//   S = ( U    )
//       ( M  L )
//
// where U is upper triangular of order m = (n+kd)/2, and L is lower
// triangular of order n-m.
func Dpbstf(uplo byte, n, kd *int, ab *mat.Matrix, ldab, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, km, m int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*ldab) < (*kd)+1 {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPBSTF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	kld = maxint(1, (*ldab)-1)

	//     Set the splitting point m.
	m = ((*n) + (*kd)) / 2

	if upper {
		//        Factorize A(m+1:n,m+1:n) as L**T*L, and update A(1:m,1:m).
		for j = (*n); j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get((*kd)+1-1, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set((*kd)+1-1, j-1, ajj)
			km = minint(j-1, *kd)

			//           Compute elements j-km:j-1 of the j-th column and update the
			//           the leading submatrix within the band.
			goblas.Dscal(km, one/ajj, ab.Vector((*kd)+1-km-1, j-1), 1)
			err = goblas.Dsyr(Upper, km, -one, ab.Vector((*kd)+1-km-1, j-1), 1, ab.OffIdx((*kd)+1-1+(j-km-1)*(*ldab)).UpdateRows(kld), kld)
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**T*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get((*kd)+1-1, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set((*kd)+1-1, j-1, ajj)
			km = minint(*kd, m-j)

			//           Compute elements j+1:j+km of the j-th row and update the
			//           trailing submatrix within the band.
			if km > 0 {
				goblas.Dscal(km, one/ajj, ab.Vector((*kd)-1, j+1-1), kld)
				err = goblas.Dsyr(Upper, km, -one, ab.Vector((*kd)-1, j+1-1), kld, ab.OffIdx((*kd)+1-1+(j+1-1)*(*ldab)).UpdateRows(kld), kld)
			}
		}
	} else {
		//        Factorize A(m+1:n,m+1:n) as L**T*L, and update A(1:m,1:m).
		for j = (*n); j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)
			km = minint(j-1, *kd)

			//           Compute elements j-km:j-1 of the j-th row and update the
			//           trailing submatrix within the band.
			goblas.Dscal(km, one/ajj, ab.Vector(km+1-1, j-km-1), kld)
			err = goblas.Dsyr(Lower, km, -one, ab.Vector(km+1-1, j-km-1), kld, ab.OffIdx(0+(j-km-1)*(*ldab)).UpdateRows(kld), kld)
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**T*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)
			km = minint(*kd, m-j)

			//           Compute elements j+1:j+km of the j-th column and update the
			//           trailing submatrix within the band.
			if km > 0 {
				goblas.Dscal(km, one/ajj, ab.Vector(1, j-1), 1)
				err = goblas.Dsyr(Lower, km, -one, ab.Vector(1, j-1), 1, ab.OffIdx(0+(j+1-1)*(*ldab)).UpdateRows(kld), kld)
			}
		}
	}
	return

label50:
	;
	(*info) = j
}
