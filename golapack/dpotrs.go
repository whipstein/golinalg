package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotrs solves a system of linear equations A*X = B with a symmetric
// positive definite matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by DPOTRF.
func Dpotrs(uplo byte, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, info *int) {
	var upper bool

	one := 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T *U.
		//
		//        Solve U**T *X = B, overwriting B with X.
		goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, n, nrhs, &one, a, lda, b, ldb)

		//        Solve U*X = B, overwriting B with X.
		goblas.Dtrsm(mat.Left, mat.Upper, mat.NoTrans, mat.NonUnit, n, nrhs, &one, a, lda, b, ldb)
	} else {
		//        Solve A*X = B where A = L*L**T.
		//
		//        Solve L*X = B, overwriting B with X.
		goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.NonUnit, n, nrhs, &one, a, lda, b, ldb)

		//        Solve L**T *X = B, overwriting B with X.
		goblas.Dtrsm(mat.Left, mat.Lower, mat.Trans, mat.NonUnit, n, nrhs, &one, a, lda, b, ldb)
	}
}
