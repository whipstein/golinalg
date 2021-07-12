package golapack

import (
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
func Dpotrf(uplo byte, n *int, a *mat.Matrix, lda, info *int) {
	var upper bool
	var one float64
	var j, jb, nb int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPOTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DPOTRF"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code.
		Dpotrf2(uplo, n, a, lda, info)
	} else {
		//        Use blocked code.
		if upper {
			//           Compute the Cholesky factorization A = U**T*U.
			for j = 1; j <= (*n); j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, (*n)-j+1)
				err = goblas.Dsyrk(mat.Upper, mat.Trans, jb, j-1, -one, a.Off(0, j-1), one, a.Off(j-1, j-1))
				Dpotrf2('U', &jb, a.Off(j-1, j-1), lda, info)
				if (*info) != 0 {
					goto label30
				}
				if j+jb <= (*n) {
					//                 Compute the current block row.
					err = goblas.Dgemm(mat.Trans, mat.NoTrans, jb, (*n)-j-jb+1, j-1, -one, a.Off(0, j-1), a.Off(0, j+jb-1), one, a.Off(j-1, j+jb-1))
					err = goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, jb, (*n)-j-jb+1, one, a.Off(j-1, j-1), a.Off(j-1, j+jb-1))
				}
			}

		} else {
			//           Compute the Cholesky factorization A = L*L**T.
			for j = 1; j <= (*n); j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, (*n)-j+1)
				err = goblas.Dsyrk(mat.Lower, mat.NoTrans, jb, j-1, -one, a.Off(j-1, 0), one, a.Off(j-1, j-1))
				Dpotrf2('L', &jb, a.Off(j-1, j-1), lda, info)
				if (*info) != 0 {
					goto label30
				}
				if j+jb <= (*n) {
					//                 Compute the current block column.
					err = goblas.Dgemm(mat.NoTrans, mat.Trans, (*n)-j-jb+1, jb, j-1, -one, a.Off(j+jb-1, 0), a.Off(j-1, 0), one, a.Off(j+jb-1, j-1))
					err = goblas.Dtrsm(mat.Right, mat.Lower, mat.Trans, mat.NonUnit, (*n)-j-jb+1, jb, one, a.Off(j-1, j-1), a.Off(j+jb-1, j-1))
				}
			}
		}
	}
	goto label40

label30:
	;
	(*info) = (*info) + j - 1

label40:
}
