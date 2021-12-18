package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zspsvx uses the diagonal pivoting factorization A = U*D*U**T or
// A = L*D*L**T to compute the solution to a complex system of linear
// equations A * X = B, where A is an N-by-N symmetric matrix stored
// in packed format and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zspsvx(fact byte, uplo mat.MatUplo, n, nrhs int, ap, afp *mat.CVector, ipiv *[]int, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (rcond float64, info int, err error) {
	var nofact bool
	var anorm, zero float64

	zero = 0.0

	//     Test the input parameters.
	nofact = fact == 'N'
	if !nofact && fact != 'F' {
		err = fmt.Errorf("!nofact && fact != 'F': fact='%c'", fact)
	} else if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
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
		gltest.Xerbla2("Zspsvx", err)
		return
	}

	if nofact {
		//        Compute the factorization A = U*D*U**T or A = L*D*L**T.
		afp.Copy(n*(n+1)/2, ap, 1, 1)
		if info, err = Zsptrf(uplo, n, afp, ipiv); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Zlansp('I', uplo, n, ap, rwork)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Zspcon(uplo, n, afp, ipiv, anorm, work); err != nil {
		panic(err)
	}

	//     Compute the solution vectors X.
	Zlacpy(Full, n, nrhs, b, x)
	if err = Zsptrs(uplo, n, nrhs, afp, ipiv, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	if err = Zsprfs(uplo, n, nrhs, ap, afp, ipiv, b, x, ferr, berr, work, rwork); err != nil {
		panic(err)
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
