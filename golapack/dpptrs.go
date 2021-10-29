package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpptrs solves a system of linear equations A*X = B with a symmetric
// positive definite matrix A in packed storage using the Cholesky
// factorization A = U**T*U or A = L*L**T computed by DPPTRF.
func Dpptrs(uplo mat.MatUplo, n, nrhs int, ap *mat.Vector, b *mat.Matrix) (err error) {
	var upper bool
	var i int

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpptrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T * U.
		for i = 1; i <= nrhs; i++ {
			//           Solve U**T *X = B, overwriting B with X.
			if err = goblas.Dtpsv(Upper, Trans, NonUnit, n, ap, b.Vector(0, i-1, 1)); err != nil {
				panic(err)
			}

			//           Solve U*X = B, overwriting B with X.
			if err = goblas.Dtpsv(Upper, NoTrans, NonUnit, n, ap, b.Vector(0, i-1, 1)); err != nil {
				panic(err)
			}
		}
	} else {
		//        Solve A*X = B where A = L * L**T.
		for i = 1; i <= nrhs; i++ {
			//           Solve L*Y = B, overwriting B with X.
			if err = goblas.Dtpsv(Lower, NoTrans, NonUnit, n, ap, b.Vector(0, i-1, 1)); err != nil {
				panic(err)
			}

			//           Solve L**T *X = Y, overwriting B with X.
			if err = goblas.Dtpsv(Lower, Trans, NonUnit, n, ap, b.Vector(0, i-1, 1)); err != nil {
				panic(err)
			}
		}
	}

	return
}
