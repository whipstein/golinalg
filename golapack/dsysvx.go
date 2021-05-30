package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// DSYSVX uses the diagonal pivoting factorization to compute the
// solution to a real system of linear equations A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dsysvx(fact, uplo byte, n, nrhs *int, a *mat.Matrix, lda *int, af *mat.Matrix, ldaf *int, ipiv *[]int, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, rcond *float64, ferr, berr, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, nofact bool
	var anorm, zero float64
	var lwkopt, nb int

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	nofact = fact == 'N'
	lquery = ((*lwork) == -1)
	if !nofact && fact != 'F' {
		(*info) = -1
	} else if uplo != 'U' && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldaf) < maxint(1, *n) {
		(*info) = -8
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -11
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -13
	} else if (*lwork) < maxint(1, 3*(*n)) && !lquery {
		(*info) = -18
	}
	//
	if (*info) == 0 {
		lwkopt = maxint(1, 3*(*n))
		if nofact {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRF"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
			lwkopt = maxint(lwkopt, (*n)*nb)
		}
		work.Set(0, float64(lwkopt))
	}
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYSVX"), -(*info))
		return
	} else if lquery {
		return
	}

	if nofact {
		//        Compute the factorization A = U*D*U**T or A = L*D*L**T.
		Dlacpy(uplo, n, n, a, lda, af, ldaf)
		Dsytrf(uplo, n, af, ldaf, ipiv, work, lwork, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Dlansy('I', uplo, n, a, lda, work)

	//     Compute the reciprocal of the condition number of A.
	Dsycon(uplo, n, af, ldaf, ipiv, &anorm, rcond, work, iwork, info)

	//     Compute the solution vectors X.
	Dlacpy('F', n, nrhs, b, ldb, x, ldx)
	Dsytrs(uplo, n, nrhs, af, ldaf, ipiv, x, ldx, info)

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	Dsyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}

	work.Set(0, float64(lwkopt))
}