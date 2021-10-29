package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpotrf computes the Cholesky factorization of a real symmetric
// positive definite matrix A.
//
// The factorization has the form
//    A = U**T * U,  if UPLO = 'U', or
//    A = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the block version of the algorithm, calling Level 3 BLAS.
func Dpotrf(uplo mat.MatUplo, n int, a *mat.Matrix) (info int, err error) {
	var upper bool
	var one float64
	var j, jb, nb int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpotrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(1, "Dpotrf", []byte{uplo.Byte()}, n, -1, -1, -1)
	if nb <= 1 || nb >= n {
		//        Use unblocked code.
		if info, err = Dpotrf2(uplo, n, a); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code.
		if upper {
			//           Compute the Cholesky factorization A = U**T*U.
			for j = 1; j <= n; j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, n-j+1)
				err = goblas.Dsyrk(mat.Upper, mat.Trans, jb, j-1, -one, a.Off(0, j-1), one, a.Off(j-1, j-1))
				if info, err = Dpotrf2(Upper, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					goto label30
				}
				if j+jb <= n {
					//                 Compute the current block row.
					err = goblas.Dgemm(mat.Trans, mat.NoTrans, jb, n-j-jb+1, j-1, -one, a.Off(0, j-1), a.Off(0, j+jb-1), one, a.Off(j-1, j+jb-1))
					err = goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, jb, n-j-jb+1, one, a.Off(j-1, j-1), a.Off(j-1, j+jb-1))
				}
			}

		} else {
			//           Compute the Cholesky factorization A = L*L**T.
			for j = 1; j <= n; j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, n-j+1)
				err = goblas.Dsyrk(mat.Lower, mat.NoTrans, jb, j-1, -one, a.Off(j-1, 0), one, a.Off(j-1, j-1))
				if info, err = Dpotrf2(Lower, jb, a.Off(j-1, j-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					goto label30
				}
				if j+jb <= n {
					//                 Compute the current block column.
					err = goblas.Dgemm(mat.NoTrans, mat.Trans, n-j-jb+1, jb, j-1, -one, a.Off(j+jb-1, 0), a.Off(j-1, 0), one, a.Off(j+jb-1, j-1))
					err = goblas.Dtrsm(mat.Right, mat.Lower, mat.Trans, mat.NonUnit, n-j-jb+1, jb, one, a.Off(j-1, j-1), a.Off(j+jb-1, j-1))
				}
			}
		}
	}
	goto label40

label30:
	;
	info = info + j - 1

label40:

	return
}
