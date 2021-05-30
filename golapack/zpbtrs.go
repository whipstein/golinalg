package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zpbtrs solves a system of linear equations A*X = B with a Hermitian
// positive definite band matrix A using the Cholesky factorization
// A = U**H *U or A = L*L**H computed by ZPBTRF.
func Zpbtrs(uplo byte, n, kd, nrhs *int, ab *mat.CMatrix, ldab *int, b *mat.CMatrix, ldb, info *int) {
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
		gltest.Xerbla([]byte("ZPBTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**H *U.
		for j = 1; j <= (*nrhs); j++ {
			//           Solve U**H *X = B, overwriting B with X.
			goblas.Ztbsv(Upper, ConjTrans, NonUnit, n, kd, ab, ldab, b.CVector(0, j-1), func() *int { y := 1; return &y }())

			//           Solve U*X = B, overwriting B with X.
			goblas.Ztbsv(Upper, NoTrans, NonUnit, n, kd, ab, ldab, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
	} else {
		//        Solve A*X = B where A = L*L**H.
		for j = 1; j <= (*nrhs); j++ {
			//           Solve L*X = B, overwriting B with X.
			goblas.Ztbsv(Lower, NoTrans, NonUnit, n, kd, ab, ldab, b.CVector(0, j-1), func() *int { y := 1; return &y }())

			//           Solve L**H *X = B, overwriting B with X.
			goblas.Ztbsv(Lower, ConjTrans, NonUnit, n, kd, ab, ldab, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
	}
}