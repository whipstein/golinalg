package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dptsvx uses the factorization A = L*D*L**T to compute the solution
// to a real system of linear equations A*X = B, where A is an N-by-N
// symmetric positive definite tridiagonal matrix and X and B are
// N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dptsvx(fact byte, n, nrhs int, d, e, df, ef *mat.Vector, b, x *mat.Matrix, ferr, berr, work *mat.Vector) (rcond float64, info int, err error) {
	var nofact bool
	var anorm, zero float64

	zero = 0.0

	//     Test the input parameters.
	nofact = fact == 'N'
	if !nofact && fact != 'F' {
		err = fmt.Errorf("!nofact && fact != 'F': fact='%c'", fact)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if x.Rows < max(1, n) {
		err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dptsvx", err)
		return
	}

	if nofact {
		//        Compute the L*D*L**T (or U**T*D*U) factorization of A.
		df.Copy(n, d, 1, 1)
		if n > 1 {
			ef.Copy(n-1, e, 1, 1)
		}
		if info, err = Dpttrf(n, df, ef); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Dlanst('1', n, d, e)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Dptcon(n, df, ef, anorm, work); err != nil {
		panic(err)
	}

	//     Compute the solution vectors X.
	Dlacpy(Full, n, nrhs, b, x)
	if err = Dpttrs(n, nrhs, df, ef, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	if err = Dptrfs(n, nrhs, d, e, df, ef, b, x, ferr, berr, work); err != nil {
		panic(err)
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
