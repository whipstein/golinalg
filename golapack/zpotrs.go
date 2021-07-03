package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotrs solves a system of linear equations A*X = B with a Hermitian
// positive definite matrix A using the Cholesky factorization
// A = U**H * U or A = L * L**H computed by ZPOTRF.
func Zpotrs(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var one complex128
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZPOTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**H *U.
		//
		//        Solve U**H *X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Upper, ConjTrans, NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve U*X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)
	} else {
		//        Solve A*X = B where A = L*L**H.
		//
		//        Solve L*X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve L**H *X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Lower, ConjTrans, NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)
	}
}
