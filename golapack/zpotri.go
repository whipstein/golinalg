package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotri computes the inverse of a complex Hermitian positive definite
// matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
// computed by ZPOTRF.
func Zpotri(uplo mat.MatUplo, n int, a *mat.CMatrix) (info int, err error) {
	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpotri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	if info, err = Ztrtri(uplo, NonUnit, n, a); err != nil {
		panic(err)
	}
	if info > 0 {
		return
	}

	//     Form inv(U) * inv(U)**H or inv(L)**H * inv(L).
	if err = Zlauum(uplo, n, a); err != nil {
		panic(err)
	}

	return
}
