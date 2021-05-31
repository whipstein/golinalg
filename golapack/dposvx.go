package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DPOSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
// compute the solution to a real system of linear equations
//    A * X = B,
// where A is an N-by-N symmetric positive definite matrix and X and B
// are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dposvx(fact, uplo byte, n, nrhs *int, a *mat.Matrix, lda *int, af *mat.Matrix, ldaf *int, equed *byte, s *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, rcond *float64, ferr, berr, work *mat.Vector, iwork *[]int, info *int) {
	var equil, nofact, rcequ bool
	var amax, anorm, bignum, one, scond, smax, smin, smlnum, zero float64
	var i, infequ, j int

	zero = 0.0
	one = 1.0

	(*info) = 0
	nofact = fact == 'N'
	equil = fact == 'E'
	if nofact || equil {
		(*equed) = 'N'
		rcequ = false
	} else {
		rcequ = *equed == 'Y'
		smlnum = Dlamch(SafeMinimum)
		bignum = one / smlnum
	}

	//     Test the input parameters.
	if !nofact && !equil && fact != 'F' {
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
	} else if fact == 'F' && !(rcequ || *equed == 'N') {
		(*info) = -9
	} else {
		if rcequ {
			smin = bignum
			smax = zero
			for j = 1; j <= (*n); j++ {
				smin = minf64(smin, s.Get(j-1))
				smax = maxf64(smax, s.Get(j-1))
			}
			if smin <= zero {
				(*info) = -10
			} else if (*n) > 0 {
				scond = maxf64(smin, smlnum) / minf64(smax, bignum)
			} else {
				scond = one
			}
		}
		if (*info) == 0 {
			if (*ldb) < maxint(1, *n) {
				(*info) = -12
			} else if (*ldx) < maxint(1, *n) {
				(*info) = -14
			}
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOSVX"), -(*info))
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		Dpoequ(n, a, lda, s, &scond, &amax, &infequ)
		if infequ == 0 {
			//           Equilibrate the matrix.
			Dlaqsy(uplo, n, a, lda, s, &scond, &amax, equed)
			rcequ = *equed == 'Y'
		}
	}

	//     Scale the right hand side.
	if rcequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				b.Set(i-1, j-1, s.Get(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the Cholesky factorization A = U**T *U or A = L*L**T.
		Dlacpy(uplo, n, n, a, lda, af, ldaf)
		Dpotrf(uplo, n, af, ldaf, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Dlansy('1', uplo, n, a, lda, work)

	//     Compute the reciprocal of the condition number of A.
	Dpocon(uplo, n, af, ldaf, &anorm, rcond, work, iwork, info)

	//     Compute the solution matrix X.
	Dlacpy('F', n, nrhs, b, ldb, x, ldx)
	Dpotrs(uplo, n, nrhs, af, ldaf, x, ldx, info)

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	Dporfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr, work, iwork, info)

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if rcequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				x.Set(i-1, j-1, s.Get(i-1)*x.Get(i-1, j-1))
			}
		}
		for j = 1; j <= (*nrhs); j++ {
			ferr.Set(j-1, ferr.Get(j-1)/scond)
		}
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}
