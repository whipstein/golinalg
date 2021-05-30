package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zpbstf computes a split Cholesky factorization of a complex
// Hermitian positive definite band matrix A.
//
// This routine is designed to be used in conjunction with ZHBGST.
//
// The factorization has the form  A = S**H*S  where S is a band matrix
// of the same bandwidth as A and the following structure:
//
//   S = ( U    )
//       ( M  L )
//
// where U is upper triangular of order m = (n+kd)/2, and L is lower
// triangular of order n-m.
func Zpbstf(uplo byte, n, kd *int, ab *mat.CMatrix, ldab, info *int) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, km, m int

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
		gltest.Xerbla([]byte("ZPBSTF"), -(*info))
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
		//        Factorize A(m+1:n,m+1:n) as L**H*L, and update A(1:m,1:m).
		for j = (*n); j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.GetRe((*kd)+1-1, j-1)
			if ajj <= zero {
				ab.SetRe((*kd)+1-1, j-1, ajj)
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe((*kd)+1-1, j-1, ajj)
			km = minint(j-1, *kd)

			//           Compute elements j-km:j-1 of the j-th column and update the
			//           the leading submatrix within the band.
			goblas.Zdscal(&km, toPtrf64(one/ajj), ab.CVector((*kd)+1-km-1, j-1), func() *int { y := 1; return &y }())
			goblas.Zher(Upper, &km, toPtrf64(-one), ab.CVector((*kd)+1-km-1, j-1), func() *int { y := 1; return &y }(), ab.Off((*kd)+1-1, j-km-1).UpdateRows(kld), &kld)
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**H*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.GetRe((*kd)+1-1, j-1)
			if ajj <= zero {
				ab.SetRe((*kd)+1-1, j-1, ajj)
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe((*kd)+1-1, j-1, ajj)
			km = minint(*kd, m-j)

			//           Compute elements j+1:j+km of the j-th row and update the
			//           trailing submatrix within the band.
			if km > 0 {
				goblas.Zdscal(&km, toPtrf64(one/ajj), ab.CVector((*kd)-1, j+1-1), &kld)
				Zlacgv(&km, ab.CVector((*kd)-1, j+1-1), &kld)
				goblas.Zher(Upper, &km, toPtrf64(-one), ab.CVector((*kd)-1, j+1-1), &kld, ab.Off((*kd)+1-1, j+1-1).UpdateRows(kld), &kld)
				Zlacgv(&km, ab.CVector((*kd)-1, j+1-1), &kld)
			}
		}
	} else {
		//        Factorize A(m+1:n,m+1:n) as L**H*L, and update A(1:m,1:m).
		for j = (*n); j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.GetRe(0, j-1)
			if ajj <= zero {
				ab.SetRe(0, j-1, ajj)
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe(0, j-1, ajj)
			km = minint(j-1, *kd)

			//           Compute elements j-km:j-1 of the j-th row and update the
			//           trailing submatrix within the band.
			goblas.Zdscal(&km, toPtrf64(one/ajj), ab.CVector(km+1-1, j-km-1), &kld)
			Zlacgv(&km, ab.CVector(km+1-1, j-km-1), &kld)
			goblas.Zher(Lower, &km, toPtrf64(-one), ab.CVector(km+1-1, j-km-1), &kld, ab.Off(0, j-km-1).UpdateRows(kld), &kld)
			Zlacgv(&km, ab.CVector(km+1-1, j-km-1), &kld)
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**H*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.GetRe(0, j-1)
			if ajj <= zero {
				ab.SetRe(0, j-1, ajj)
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.SetRe(0, j-1, ajj)
			km = minint(*kd, m-j)

			//           Compute elements j+1:j+km of the j-th column and update the
			//           trailing submatrix within the band.
			if km > 0 {
				goblas.Zdscal(&km, toPtrf64(one/ajj), ab.CVector(1, j-1), func() *int { y := 1; return &y }())
				goblas.Zher(Lower, &km, toPtrf64(-one), ab.CVector(1, j-1), func() *int { y := 1; return &y }(), ab.Off(0, j+1-1).UpdateRows(kld), &kld)
			}
		}
	}
	return

label50:
	;
	(*info) = j
}
