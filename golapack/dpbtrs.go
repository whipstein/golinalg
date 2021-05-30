package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dpbtrs solves a system of linear equations A*X = B with a symmetric
// positive definite band matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by DPBTRF.
func Dpbtrs(uplo byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, b *mat.Matrix, ldb, info *int) {
	var upper bool
	var j int

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*ldab) < (*kd)+1 {
		(*info) = -6
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPBTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T *U.
		for j = 1; j <= (*nrhs); j++ {
			//           Solve U**T *X = B, overwriting B with X.
			goblas.Dtbsv(mat.Upper, mat.Trans, mat.NonUnit, n, kd, ab, ldab, b.Vector(0, j-1), toPtr(1))

			//           Solve U*X = B, overwriting B with X.
			goblas.Dtbsv(mat.Upper, mat.NoTrans, mat.NonUnit, n, kd, ab, ldab, b.Vector(0, j-1), toPtr(1))
		}
	} else {
		//        Solve A*X = B where A = L*L**T.
		for j = 1; j <= (*nrhs); j++ {
			//           Solve L*X = B, overwriting B with X.
			goblas.Dtbsv(mat.Lower, mat.NoTrans, mat.NonUnit, n, kd, ab, ldab, b.Vector(0, j-1), toPtr(1))

			//           Solve L**T *X = B, overwriting B with X.
			goblas.Dtbsv(mat.Lower, mat.Trans, mat.NonUnit, n, kd, ab, ldab, b.Vector(0, j-1), toPtr(1))
		}
	}
}
