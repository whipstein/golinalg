package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpptrs solves a system of linear equations A*X = B with a symmetric
// positive definite matrix A in packed storage using the Cholesky
// factorization A = U**T*U or A = L*L**T computed by DPPTRF.
func Dpptrs(uplo byte, n, nrhs *int, ap *mat.Vector, b *mat.Matrix, ldb, info *int) {
	var upper bool
	var i int
	var err error
	_ = err

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
		gltest.Xerbla([]byte("DPPTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T * U.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve U**T *X = B, overwriting B with X.
			err = goblas.Dtpsv(mat.Upper, mat.Trans, mat.NonUnit, *n, ap, b.Vector(0, i-1), 1)

			//           Solve U*X = B, overwriting B with X.
			err = goblas.Dtpsv(mat.Upper, mat.NoTrans, mat.NonUnit, *n, ap, b.Vector(0, i-1), 1)
		}
	} else {
		//        Solve A*X = B where A = L * L**T.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve L*Y = B, overwriting B with X.
			err = goblas.Dtpsv(mat.Lower, mat.NoTrans, mat.NonUnit, *n, ap, b.Vector(0, i-1), 1)

			//           Solve L**T *X = Y, overwriting B with X.
			err = goblas.Dtpsv(mat.Lower, mat.Trans, mat.NonUnit, *n, ap, b.Vector(0, i-1), 1)
		}
	}
}
