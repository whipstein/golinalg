package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zppsvx uses the Cholesky factorization A = U**H * U or A = L * L**H to
// compute the solution to a complex system of linear equations
//    A * X = B,
// where A is an N-by-N Hermitian positive definite matrix stored in
// packed format and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zppsvx(fact byte, uplo mat.MatUplo, n, nrhs int, ap, afp *mat.CVector, equed byte, s *mat.Vector, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (equedOut byte, rcond float64, info int, err error) {
	var equil, nofact, rcequ bool
	var amax, anorm, bignum, one, scond, smax, smin, smlnum, zero float64
	var i, infequ, j int

	zero = 0.0
	one = 1.0
	equedOut = equed

	nofact = fact == 'N'
	equil = fact == 'E'
	if nofact || equil {
		equedOut = 'N'
		rcequ = false
	} else {
		rcequ = equedOut == 'Y'
		smlnum = Dlamch(SafeMinimum)
		bignum = one / smlnum
	}

	//     Test the input parameters.
	if !nofact && !equil && fact != 'F' {
		err = fmt.Errorf("!nofact && !equil && fact != 'F': fact='%c'", fact)
	} else if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if fact == 'F' && !(rcequ || equedOut == 'N') {
		err = fmt.Errorf("fact == 'F' && !(rcequ || equed == 'N'): fact='%c', equed='%c', rcequ=%v", fact, equedOut, rcequ)
	} else {
		if rcequ {
			smin = bignum
			smax = zero
			for j = 1; j <= n; j++ {
				smin = math.Min(smin, s.Get(j-1))
				smax = math.Max(smax, s.Get(j-1))
			}
			if smin <= zero {
				err = fmt.Errorf("smin <= zero: smin=%v", smin)
			} else if n > 0 {
				scond = math.Max(smin, smlnum) / math.Min(smax, bignum)
			} else {
				scond = one
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

	if err != nil {
		gltest.Xerbla2("Zppsvx", err)
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		if scond, amax, infequ, err = Zppequ(uplo, n, ap, s); err != nil {
			panic(err)
		}
		if infequ == 0 {
			//           Equilibrate the matrix.
			equedOut = Zlaqhp(uplo, n, ap, s, scond, amax)
			rcequ = equedOut == 'Y'
		}
	}

	//     Scale the right-hand side.
	if rcequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, s.GetCmplx(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the Cholesky factorization A = U**H * U or A = L * L**H.
		afp.Copy(n*(n+1)/2, ap, 1, 1)
		if info, err = Zpptrf(uplo, n, afp); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlanhp('I', uplo, n, ap, rwork)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Zppcon(uplo, n, afp, anorm, work, rwork); err != nil {
		panic(err)
	}

	//     Compute the solution matrix X.
	Zlacpy(Full, n, nrhs, b, x)
	if err = Zpptrs(uplo, n, nrhs, afp, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	if err = Zpprfs(uplo, n, nrhs, ap, afp, b, x, ferr, berr, work, rwork); err != nil {
		panic(err)
	}

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if rcequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				x.Set(i-1, j-1, s.GetCmplx(i-1)*x.Get(i-1, j-1))
			}
		}
		for j = 1; j <= nrhs; j++ {
			ferr.Set(j-1, ferr.Get(j-1)/scond)
		}
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
