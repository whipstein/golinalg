package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsysvx uses the diagonal pivoting factorization to compute the
// solution to a real system of linear equations A * X = B,
// where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
// matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Dsysvx(fact byte, uplo mat.MatUplo, n, nrhs int, a, af *mat.Matrix, ipiv *[]int, b, x *mat.Matrix, ferr, berr, work *mat.Vector, lwork int, iwork *[]int) (rcond float64, info int, err error) {
	var lquery, nofact bool
	var anorm, zero float64
	var lwkopt, nb int

	zero = 0.0

	//     Test the input parameters.
	nofact = fact == 'N'
	lquery = (lwork == -1)
	if !nofact && fact != 'F' {
		err = fmt.Errorf("!nofact && fact != 'F': fact='%c'", fact)
	} else if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if af.Rows < max(1, n) {
		err = fmt.Errorf("af.Rows < max(1, n): af.Rows=%v, n=%v", af.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if x.Rows < max(1, n) {
		err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
	} else if lwork < max(1, 3*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 3*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err == nil {
		lwkopt = max(1, 3*n)
		if nofact {
			nb = Ilaenv(1, "Dsytrf", []byte{uplo.Byte()}, n, -1, -1, -1)
			lwkopt = max(lwkopt, n*nb)
		}
		work.Set(0, float64(lwkopt))
	}
	//
	if err != nil {
		gltest.Xerbla2("Dsysvx", err)
		return
	} else if lquery {
		return
	}

	if nofact {
		//        Compute the factorization A = U*D*U**T or A = L*D*L**T.
		Dlacpy(uplo, n, n, a, af)
		if info, err = Dsytrf(uplo, n, af, ipiv, work, lwork); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A.
	anorm = Dlansy('I', uplo, n, a, work)

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Dsycon(uplo, n, af, ipiv, anorm, work, iwork); err != nil {
		panic(err)
	}

	//     Compute the solution vectors X.
	Dlacpy(Full, n, nrhs, b, x)
	if err = Dsytrs(uplo, n, nrhs, af, ipiv, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solutions and
	//     compute error bounds and backward error estimates for them.
	if info, err = Dsyrfs(uplo, n, nrhs, a, af, ipiv, b, x, ferr, berr, work, iwork); err != nil {
		panic(err)
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	work.Set(0, float64(lwkopt))

	return
}
