package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbtrs solves a system of linear equations A*X = B with a symmetric
// positive definite band matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by DPBTRF.
func Dpbtrs(uplo mat.MatUplo, n, kd, nrhs int, ab, b *mat.Matrix) (err error) {
	var upper bool
	var j int

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpbtrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B where A = U**T *U.
		for j = 1; j <= nrhs; j++ {
			//           Solve U**T *X = B, overwriting B with X.
			if err = b.Off(0, j-1).Vector().Tbsv(Upper, Trans, NonUnit, n, kd, ab, 1); err != nil {
				panic(err)
			}

			//           Solve U*X = B, overwriting B with X.
			if err = b.Off(0, j-1).Vector().Tbsv(Upper, NoTrans, NonUnit, n, kd, ab, 1); err != nil {
				panic(err)
			}
		}
	} else {
		//        Solve A*X = B where A = L*L**T.
		for j = 1; j <= nrhs; j++ {
			//           Solve L*X = B, overwriting B with X.
			if err = b.Off(0, j-1).Vector().Tbsv(Lower, NoTrans, NonUnit, n, kd, ab, 1); err != nil {
				panic(err)
			}

			//           Solve L**T *X = B, overwriting B with X.
			if err = b.Off(0, j-1).Vector().Tbsv(Lower, Trans, NonUnit, n, kd, ab, 1); err != nil {
				panic(err)
			}
		}
	}

	return
}
