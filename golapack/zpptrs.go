package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpptrs solves a system of linear equations A*X = B with a Hermitian
// positive definite matrix A in packed storage using the Cholesky
// factorization A = U**H * U or A = L * L**H computed by ZPPTRF.
func Zpptrs(uplo byte, n, nrhs *int, ap *mat.CVector, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var i int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPPTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**H * U.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve U**H *X = B, overwriting B with X.
			goblas.Ztpsv(Upper, ConjTrans, NonUnit, n, ap, b.CVector(0, i-1), func() *int { y := 1; return &y }())

			//           Solve U*X = B, overwriting B with X.
			goblas.Ztpsv(Upper, NoTrans, NonUnit, n, ap, b.CVector(0, i-1), func() *int { y := 1; return &y }())
		}
	} else {
		//        Solve A*X = B where A = L * L**H.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve L*Y = B, overwriting B with X.
			goblas.Ztpsv(Lower, NoTrans, NonUnit, n, ap, b.CVector(0, i-1), func() *int { y := 1; return &y }())

			//           Solve L**H *X = Y, overwriting B with X.
			goblas.Ztpsv(Lower, ConjTrans, NonUnit, n, ap, b.CVector(0, i-1), func() *int { y := 1; return &y }())
		}
	}
}
