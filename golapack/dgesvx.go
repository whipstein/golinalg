package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvx uses the LU factorization to compute the solution to a real
// system of linear equations
//    A * X = B,
// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dgesvx(fact byte, trans mat.MatTrans, n, nrhs int, a, af *mat.Matrix, ipiv *[]int, equed byte, r, c *mat.Vector, b, x *mat.Matrix, ferr, berr, work *mat.Vector, iwork *[]int) (equedOut byte, rcond float64, info int, err error) {
	var colequ, equil, nofact, notran, rowequ bool
	var norm byte
	var amax, anorm, bignum, colcnd, one, rcmax, rcmin, rowcnd, rpvgrw, smlnum, zero float64
	var i, infequ, j int

	zero = 0.0
	one = 1.0
	equedOut = equed

	nofact = fact == 'N'
	equil = fact == 'E'
	notran = trans == NoTrans
	if nofact || equil {
		equedOut = 'N'
		rowequ = false
		colequ = false
	} else {
		rowequ = equedOut == 'R' || equedOut == 'B'
		colequ = equedOut == 'C' || equedOut == 'B'
		smlnum = Dlamch(SafeMinimum)
		bignum = one / smlnum
	}

	//     Test the input parameters.
	if !nofact && !equil && fact != 'F' {
		err = fmt.Errorf("!nofact && !equil && fact != 'F': fact='%c'", fact)
	} else if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if af.Rows < max(1, n) {
		err = fmt.Errorf("af.Rows < max(1, n): af.Rows=%v, n=%v", af.Rows, n)
	} else if fact == 'F' && !(rowequ || colequ || equedOut == 'N') {
		err = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equedOut == 'N'): fact='%c', equed='%c'", fact, equed)
	} else {
		if rowequ {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= n; j++ {
				rcmin = math.Min(rcmin, r.Get(j-1))
				rcmax = math.Max(rcmax, r.Get(j-1))
			}
			if rcmin <= zero {
				err = fmt.Errorf("rcmin <= zero: rcmin=%v", rcmin)
			} else if n > 0 {
				rowcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
			} else {
				rowcnd = one
			}
		}
		if colequ && err == nil {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= n; j++ {
				rcmin = math.Min(rcmin, c.Get(j-1))
				rcmax = math.Max(rcmax, c.Get(j-1))
			}
			if rcmin <= zero {
				err = fmt.Errorf("rcmin <= zero: rcmin=%v", rcmin)
			} else if n > 0 {
				colcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
			} else {
				colcnd = one
			}
		}
		if err == nil {
			if b.Rows < max(1, n) {
				err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
			} else if x.Rows < max(1, n) {
				err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
			}
		}
	}
	//
	if err != nil {
		gltest.Xerbla2("Dgesvx", err)
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		if rowcnd, colcnd, amax, infequ, err = Dgeequ(n, n, a, r, c); err != nil {
			panic(err)
		}
		if infequ == 0 {
			//           Equilibrate the matrix.
			equedOut = Dlaqge(n, n, a, r, c, rowcnd, colcnd, amax)
			rowequ = equedOut == 'R' || equedOut == 'B'
			colequ = equedOut == 'C' || equedOut == 'B'
		}
	}

	//     Scale the right hand side.
	if notran {
		if rowequ {
			for j = 1; j <= nrhs; j++ {
				for i = 1; i <= n; i++ {
					b.Set(i-1, j-1, r.Get(i-1)*b.Get(i-1, j-1))
				}
			}
		}
	} else if colequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, c.Get(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the LU factorization of A.
		Dlacpy(Full, n, n, a, af)
		if info, err = Dgetrf(n, n, af, ipiv); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			//           Compute the reciprocal pivot growth factor of the
			//           leading rank-deficient INFO columns of A.
			rpvgrw = Dlantr('M', Upper, NonUnit, info, info, af, work)
			if rpvgrw == zero {
				rpvgrw = one
			} else {
				rpvgrw = Dlange('M', n, info, a, work) / rpvgrw
			}
			work.Set(0, rpvgrw)
			rcond = zero
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
	anorm = Dlange(norm, n, n, a, work)
	rpvgrw = Dlantr('M', Upper, NonUnit, n, n, af, work)
	if rpvgrw == zero {
		rpvgrw = one
	} else {
		rpvgrw = Dlange('M', n, n, a, work) / rpvgrw
	}

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Dgecon(norm, n, af, anorm, work, iwork); err != nil {
		panic(err)
	}

	//     Compute the solution matrix X.
	Dlacpy(Full, n, nrhs, b, x)
	if err = Dgetrs(trans, n, nrhs, af, *ipiv, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	if err = Dgerfs(trans, n, nrhs, a, af, *ipiv, b, x, ferr, berr, work, iwork); err != nil {
		panic(err)
	}

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if notran {
		if colequ {
			for j = 1; j <= nrhs; j++ {
				for i = 1; i <= n; i++ {
					x.Set(i-1, j-1, c.Get(i-1)*x.Get(i-1, j-1))
				}
			}
			for j = 1; j <= nrhs; j++ {
				ferr.Set(j-1, ferr.Get(j-1)/colcnd)
			}
		}
	} else if rowequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				x.Set(i-1, j-1, r.Get(i-1)*x.Get(i-1, j-1))
			}
		}
		for j = 1; j <= nrhs; j++ {
			ferr.Set(j-1, ferr.Get(j-1)/rowcnd)
		}
	}

	work.Set(0, rpvgrw)

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
