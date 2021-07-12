package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpotrf computes the Cholesky factorization of a complex Hermitian
// positive definite matrix A.
//
// The factorization has the form
//    A = U**H * U,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the block version of the algorithm, calling Level 3 BLAS.
func Zpotrf(uplo byte, n *int, a *mat.CMatrix, lda, info *int) {
	var upper bool
	var cone complex128
	var one float64
	var j, jb, nb int
	var err error
	_ = err

	one = 1.0
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZPOTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZPOTRF"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code.
		Zpotrf2(uplo, n, a, lda, info)
	} else {
		//        Use blocked code.
		if upper {
			//           Compute the Cholesky factorization A = U**H *U.
			for j = 1; j <= (*n); j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, (*n)-j+1)
				err = goblas.Zherk(Upper, ConjTrans, jb, j-1, -one, a.Off(0, j-1), one, a.Off(j-1, j-1))
				Zpotrf2('U', &jb, a.Off(j-1, j-1), lda, info)
				if (*info) != 0 {
					goto label30
				}
				if j+jb <= (*n) {
					//                 Compute the current block row.
					err = goblas.Zgemm(ConjTrans, NoTrans, jb, (*n)-j-jb+1, j-1, -cone, a.Off(0, j-1), a.Off(0, j+jb-1), cone, a.Off(j-1, j+jb-1))
					err = goblas.Ztrsm(Left, Upper, ConjTrans, NonUnit, jb, (*n)-j-jb+1, cone, a.Off(j-1, j-1), a.Off(j-1, j+jb-1))
				}
			}

		} else {
			//           Compute the Cholesky factorization A = L*L**H.
			for j = 1; j <= (*n); j += nb {
				//              Update and factorize the current diagonal block and test
				//              for non-positive-definiteness.
				jb = min(nb, (*n)-j+1)
				err = goblas.Zherk(Lower, NoTrans, jb, j-1, -one, a.Off(j-1, 0), one, a.Off(j-1, j-1))
				Zpotrf2('L', &jb, a.Off(j-1, j-1), lda, info)
				if (*info) != 0 {
					goto label30
				}
				if j+jb <= (*n) {
					//                 Compute the current block column.
					err = goblas.Zgemm(NoTrans, ConjTrans, (*n)-j-jb+1, jb, j-1, -cone, a.Off(j+jb-1, 0), a.Off(j-1, 0), cone, a.Off(j+jb-1, j-1))
					err = goblas.Ztrsm(Right, Lower, ConjTrans, NonUnit, (*n)-j-jb+1, jb, cone, a.Off(j-1, j-1), a.Off(j+jb-1, j-1))
				}
			}
		}
	}
	return

label30:
	;
	(*info) = (*info) + j - 1
}
