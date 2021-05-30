package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// DGESVX uses the LU factorization to compute the solution to a real
// system of linear equations
//    A * X = B,
// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dgesvx(fact, trans byte, n, nrhs *int, a *mat.Matrix, lda *int, af *mat.Matrix, ldaf *int, ipiv *[]int, equed *byte, r, c *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, rcond *float64, ferr, berr, work *mat.Vector, iwork *[]int, info *int) {
	var colequ, equil, nofact, notran, rowequ bool
	var norm byte
	var amax, anorm, bignum, colcnd, one, rcmax, rcmin, rowcnd, rpvgrw, smlnum, zero float64
	var i, infequ, j int

	zero = 0.0
	one = 1.0

	(*info) = 0
	nofact = fact == 'N'
	equil = fact == 'E'
	notran = trans == 'N'
	if nofact || equil {
		(*equed) = 'N'
		rowequ = false
		colequ = false
	} else {
		rowequ = *equed == 'R' || *equed == 'B'
		colequ = *equed == 'C' || *equed == 'B'
		smlnum = Dlamch(SafeMinimum)
		bignum = one / smlnum
	}

	//     Test the input parameters.
	if !nofact && !equil && fact != 'F' {
		(*info) = -1
	} else if !notran && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldaf) < maxint(1, *n) {
		(*info) = -8
	} else if fact == 'F' && !(rowequ || colequ || *equed == 'N') {
		(*info) = -10
	} else {
		if rowequ {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= (*n); j++ {
				rcmin = minf64(rcmin, r.Get(j-1))
				rcmax = maxf64(rcmax, r.Get(j-1))
			}
			if rcmin <= zero {
				(*info) = -11
			} else if (*n) > 0 {
				rowcnd = maxf64(rcmin, smlnum) / minf64(rcmax, bignum)
			} else {
				rowcnd = one
			}
		}
		if colequ && (*info) == 0 {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= (*n); j++ {
				rcmin = minf64(rcmin, c.Get(j-1))
				rcmax = maxf64(rcmax, c.Get(j-1))
			}
			if rcmin <= zero {
				(*info) = -12
			} else if (*n) > 0 {
				colcnd = maxf64(rcmin, smlnum) / minf64(rcmax, bignum)
			} else {
				colcnd = one
			}
		}
		if (*info) == 0 {
			if (*ldb) < maxint(1, *n) {
				(*info) = -14
			} else if (*ldx) < maxint(1, *n) {
				(*info) = -16
			}
		}
	}
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESVX"), -(*info))
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		Dgeequ(n, n, a, lda, r, c, &rowcnd, &colcnd, &amax, &infequ)
		if infequ == 0 {
			//           Equilibrate the matrix.
			Dlaqge(n, n, a, lda, r, c, &rowcnd, &colcnd, &amax, equed)
			rowequ = *equed == 'R' || *equed == 'B'
			colequ = *equed == 'C' || *equed == 'B'
		}
	}

	//     Scale the right hand side.
	if notran {
		if rowequ {
			for j = 1; j <= (*nrhs); j++ {
				for i = 1; i <= (*n); i++ {
					b.Set(i-1, j-1, r.Get(i-1)*b.Get(i-1, j-1))
				}
			}
		}
	} else if colequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				b.Set(i-1, j-1, c.Get(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the LU factorization of A.
		Dlacpy('F', n, n, a, lda, af, ldaf)
		Dgetrf(n, n, af, ldaf, ipiv, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			//           Compute the reciprocal pivot growth factor of the
			//           leading rank-deficient INFO columns of A.
			rpvgrw = Dlantr('M', 'U', 'N', info, info, af, ldaf, work)
			if rpvgrw == zero {
				rpvgrw = one
			} else {
				rpvgrw = Dlange('M', n, info, a, lda, work) / rpvgrw
			}
			work.Set(0, rpvgrw)
			(*rcond) = zero
			return
		}
	}

	//     Compute the norm of the matrix A and the
	//     reciprocal pivot growth factor RPVGRW.
	if notran {
		norm = '1'
	} else {
		norm = 'I'
	}
	anorm = Dlange(norm, n, n, a, lda, work)
	rpvgrw = Dlantr('M', 'U', 'N', n, n, af, ldaf, work)
	if rpvgrw == zero {
		rpvgrw = one
	} else {
		rpvgrw = Dlange('M', n, n, a, lda, work) / rpvgrw
	}

	//     Compute the reciprocal of the condition number of A.
	Dgecon(norm, n, af, ldaf, &anorm, rcond, work, iwork, info)

	//     Compute the solution matrix X.
	Dlacpy('F', n, nrhs, b, ldb, x, ldx)
	Dgetrs(trans, n, nrhs, af, ldaf, ipiv, x, ldx, info)

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	Dgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info)

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if notran {
		if colequ {
			for j = 1; j <= (*nrhs); j++ {
				for i = 1; i <= (*n); i++ {
					x.Set(i-1, j-1, c.Get(i-1)*x.Get(i-1, j-1))
				}
			}
			for j = 1; j <= (*nrhs); j++ {
				ferr.Set(j-1, ferr.Get(j-1)/colcnd)
			}
		}
	} else if rowequ {
		for j = 1; j <= (*nrhs); j++ {
			for i = 1; i <= (*n); i++ {
				x.Set(i-1, j-1, r.Get(i-1)*x.Get(i-1, j-1))
			}
		}
		for j = 1; j <= (*nrhs); j++ {
			ferr.Set(j-1, ferr.Get(j-1)/rowcnd)
		}
	}

	work.Set(0, rpvgrw)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}
}
