package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zppsvx uses the Cholesky factorization A = U**H * U or A = L * L**H to
// compute the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian positive definite matrix stored in
// packed format and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zppsvx(fact, uplo byte, n, nrhs *int, ap, afp *mat.CVector, equed *byte, s *mat.Vector, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, rcond *float64, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
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
		rcequ = (*equed) == 'Y'
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
	} else if fact == 'F' && !(rcequ || (*equed) == 'N') {
		(*info) = -7
	} else {
		if rcequ {
			smin = bignum
			smax = zero
			for j = 1; j <= (*n); j++ {
				smin = minf64(smin, s.Get(j-1))
				smax = maxf64(smax, s.Get(j-1))
			}
			if smin <= zero {
				(*info) = -8
			} else if (*n) > 0 {
				scond = maxf64(smin, smlnum) / minf64(smax, bignum)
			} else {
				scond = one
			}
		}
		if (*info) == 0 {
			if (*ldb) < maxint(1, *n) {
				(*info) = -10
			} else if (*ldx) < maxint(1, *n) {
				(*info) = -12
			}
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPPSVX"), -(*info))
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		Zppequ(uplo, n, ap, s, &scond, &amax, &infequ)
		if infequ == 0 {
			//           Equilibrate the matrix.
			Zlaqhp(uplo, n, ap, s, &scond, &amax, equed)
			rcequ = (*equed) == 'Y'
		}
	}

	//     Scale the right-hand side.
	if rcequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				b.Set(i-1, j-1, s.GetCmplx(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the Cholesky factorization A = U**H * U or A = L * L**H.
		goblas.Zcopy(toPtr((*n)*((*n)+1)/2), ap, func() *int { y := 1; return &y }(), afp, func() *int { y := 1; return &y }())
		Zpptrf(uplo, n, afp, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlanhp('I', uplo, n, ap, rwork)

	//     Compute the reciprocal of the condition number of A.
	Zppcon(uplo, n, afp, &anorm, rcond, work, rwork, info)

	//     Compute the solution matrix X.
	Zlacpy('F', n, nrhs, b, ldb, x, ldx)
	Zpptrs(uplo, n, nrhs, afp, x, ldx, info)

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	Zpprfs(uplo, n, nrhs, ap, afp, b, ldb, x, ldx, ferr, berr, work, rwork, info)

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if rcequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				x.Set(i-1, j-1, s.GetCmplx(i-1)*x.Get(i-1, j-1))
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
