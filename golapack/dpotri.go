package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotri computes the inverse of a real symmetric positive definite
// matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
// computed by DPOTRF.
func Dpotri(uplo mat.MatUplo, n int, a *mat.Matrix) (info int, err error) {
	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpotri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	if info, err = Dtrtri(uplo, NonUnit, n, a); err != nil {
		panic(err)
	}
	if info > 0 {
		return
	}

	//     Form inv(U) * inv(U)**T or inv(L)**T * inv(L).
	if err = Dlauum(uplo, n, a); err != nil {
		panic(err)
	}

	return
}
