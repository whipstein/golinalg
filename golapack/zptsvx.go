package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zptsvx uses the factorization A = L*D*L**H to compute the solution
// to a complex system of linear equations A*X = B, where A is an
// N-by-N Hermitian positive definite tridiagonal matrix and X and B
// are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zptsvx(fact byte, n, nrhs int, d *mat.Vector, e *mat.CVector, df *mat.Vector, ef *mat.CVector, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (rcond float64, info int, err error) {
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
		gltest.Xerbla2("Zptsvx", err)
		return
	}

	if nofact {
		//        Compute the L*D*L**H (or U**H*D*U) factorization of A.
		goblas.Dcopy(n, d.Off(0, 1), df.Off(0, 1))
		if n > 1 {
			goblas.Zcopy(n-1, e.Off(0, 1), ef.Off(0, 1))
		}
		if info, err = Zpttrf(n, df, ef); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlanht('1', n, d, e)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Zptcon(n, df, ef, anorm, rwork); err != nil {
		panic(err)
	}

	//     Compute the solution vectors X.
	Zlacpy(Full, n, nrhs, b, x)
	if err = Zpttrs(Lower, n, nrhs, df, ef, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	if err = Zptrfs(Lower, n, nrhs, d, e, df, ef, b, x, ferr, berr, work, rwork); err != nil {
		panic(err)
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
