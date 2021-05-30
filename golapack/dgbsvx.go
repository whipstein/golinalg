package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dgbsvx uses the LU factorization to compute the solution to a real
// system of linear equations A * X = B, A**T * X = B, or A**H * X = B,
// where A is a band matrix of order N with KL subdiagonals and KU
// superdiagonals, and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dgbsvx(fact, trans byte, n, kl, ku, nrhs *int, ab *mat.Matrix, ldab *int, afb *mat.Matrix, ldafb *int, ipiv *[]int, equed *byte, r, c *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, rcond *float64, ferr, berr, work *mat.Vector, iwork *[]int, info *int) {
	var colequ, equil, nofact, notran, rowequ bool
	var norm byte
	var amax, anorm, bignum, colcnd, one, rcmax, rcmin, rowcnd, rpvgrw, smlnum, zero float64
	var i, infequ, j, j1, j2 int

	zero = 0.0
	one = 1.0

	(*info) = 0
	nofact = fact == 'N'
	equil = fact == 'E'
	notran = trans == 'N'
	if nofact || equil {
		*equed = 'N'
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
	} else if (*kl) < 0 {
		(*info) = -4
	} else if (*ku) < 0 {
		(*info) = -5
	} else if (*nrhs) < 0 {
		(*info) = -6
	} else if (*ldab) < (*kl)+(*ku)+1 {
		(*info) = -8
	} else if (*ldafb) < 2*(*kl)+(*ku)+1 {
		(*info) = -10
	} else if fact == 'F' && !(rowequ || colequ || *equed == 'N') {
		(*info) = -12
	} else {
		if rowequ {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= (*n); j++ {
				rcmin = minf64(rcmin, r.Get(j-1))
				rcmax = maxf64(rcmax, r.Get(j-1))
			}
			if rcmin <= zero {
				(*info) = -13
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
				(*info) = -14
			} else if (*n) > 0 {
				colcnd = maxf64(rcmin, smlnum) / minf64(rcmax, bignum)
			} else {
				colcnd = one
			}
		}
		if (*info) == 0 {
			if (*ldb) < maxint(1, *n) {
				(*info) = -16
			} else if (*ldx) < maxint(1, *n) {
				(*info) = -18
			}
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGBSVX"), -(*info))
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		Dgbequ(n, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax, &infequ)
		if infequ == 0 {
			//           Equilibrate the matrix.
			Dlaqgb(n, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax, equed)
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
		//        Compute the LU factorization of the band matrix A.
		for j = 1; j <= (*n); j++ {
			j1 = maxint(j-(*ku), 1)
			j2 = minint(j+(*kl), *n)
			goblas.Dcopy(toPtr(j2-j1+1), ab.Vector((*ku)+1-j+j1-1, j-1), toPtr(1), afb.Vector((*kl)+(*ku)+1-j+j1-1, j-1), toPtr(1))
		}

		Dgbtrf(n, n, kl, ku, afb, ldafb, ipiv, info)

		//        Return if INFO is non-zero.
		if (*info) > 0 {
			//           Compute the reciprocal pivot growth factor of the
			//           leading rank-deficient INFO columns of A.
			anorm = zero
			for j = 1; j <= (*info); j++ {
				for i = maxint((*ku)+2-j, 1); i <= minint((*n)+(*ku)+1-j, (*kl)+(*ku)+1); i++ {
					anorm = maxf64(anorm, math.Abs(ab.Get(i-1, j-1)))
				}
			}
			rpvgrw = Dlantb('M', 'U', 'N', info, toPtr(minint((*info)-1, (*kl)+(*ku))), afb.Off(maxint(1, (*kl)+(*ku)+2-(*info))-1, 0), ldafb, work)
			if rpvgrw == zero {
				rpvgrw = one
			} else {
				rpvgrw = anorm / rpvgrw
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
	anorm = Dlangb(norm, n, kl, ku, ab, ldab, work)
	rpvgrw = Dlantb('M', 'U', 'N', n, toPtr((*kl)+(*ku)), afb, ldafb, work)
	if rpvgrw == zero {
		rpvgrw = one
	} else {
		rpvgrw = Dlangb('M', n, kl, ku, ab, ldab, work) / rpvgrw
	}

	//     Compute the reciprocal of the condition number of A.
	Dgbcon(norm, n, kl, ku, afb, ldafb, ipiv, &anorm, rcond, work, iwork, info)

	//     Compute the solution matrix X.
	Dlacpy('F', n, nrhs, b, ldb, x, ldx)
	Dgbtrs(trans, n, kl, ku, nrhs, afb, ldafb, ipiv, x, ldx, info)

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	Dgbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info)

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

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if (*rcond) < Dlamch(Epsilon) {
		(*info) = (*n) + 1
	}

	work.Set(0, rpvgrw)
}
