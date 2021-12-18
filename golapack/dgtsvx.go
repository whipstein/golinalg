package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgtsvx uses the LU factorization to compute the solution to a real
// system of linear equations A * X = B or A**T * X = B,
// where A is a tridiagonal matrix of order N and X and B are N-by-NRHS
// matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dgtsvx(fact byte, trans mat.MatTrans, n, nrhs int, dl, d, du, dlf, df, duf, du2 *mat.Vector, ipiv *[]int, b, x *mat.Matrix, ferr, berr, work *mat.Vector, iwork *[]int) (rcond float64, info int, err error) {
	var nofact, notran bool
	var norm byte
	var anorm, zero float64

	zero = 0.0

	nofact = fact == 'N'
	notran = trans == NoTrans
	if !nofact && fact != 'F' {
		err = fmt.Errorf("!nofact && fact != 'F': fact='%c'", fact)
	} else if !trans.IsValid() {
		err = fmt.Errorf("!trans.IsValid(): trans=%s", trans)
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
		gltest.Xerbla2("Dgtsvx", err)
		return
	}

	if nofact {
		//        Compute the LU factorization of A.
		df.Copy(n, d, 1, 1)
		if n > 1 {
			dlf.Copy(n-1, dl, 1, 1)
			duf.Copy(n-1, du, 1, 1)
		}
		if info, err = Dgttrf(n, dlf, df, duf, du2, ipiv); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	if notran {
		norm = '1'
	} else {
		norm = 'I'
	}
	anorm = Dlangt(norm, n, dl, d, du)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Dgtcon(norm, n, dlf, df, duf, du2, *ipiv, anorm, work, iwork); err != nil {
		panic(err)
	}

	//     Compute the solution vectors X.
	Dlacpy(Full, n, nrhs, b, x)
	if err = Dgttrs(trans, n, nrhs, dlf, df, duf, du2, *ipiv, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	if err = Dgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, *ipiv, b, x, ferr, berr, work, iwork); err != nil {
		panic(err)
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	return
}
