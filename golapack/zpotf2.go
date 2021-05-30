package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zpotf2 computes the Cholesky factorization of a complex Hermitian
// positive definite matrix A.
//
// The factorization has the form
//    A = U**H * U ,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zpotf2(uplo byte, n *int, a *mat.CMatrix, lda, info *int) {
	var upper bool
	var cone complex128
	var ajj, one, zero float64
	var j int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPOTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Compute the Cholesky factorization A = U**H *U.
		for j = 1; j <= (*n); j++ {
			//           Compute U(J,J) and test for non-positive-definiteness.
			ajj = a.GetRe(j-1, j-1) - real(goblas.Zdotc(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }(), a.CVector(0, j-1), func() *int { y := 1; return &y }()))
			if ajj <= zero || Disnan(int(ajj)) {
				a.SetRe(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J.
			if j < (*n) {
				Zlacgv(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(Trans, toPtr(j-1), toPtr((*n)-j), toPtrc128(-cone), a.Off(0, j+1-1), lda, a.CVector(0, j-1), func() *int { y := 1; return &y }(), &cone, a.CVector(j-1, j+1-1), lda)
				Zlacgv(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(toPtr((*n)-j), toPtrf64(one/ajj), a.CVector(j-1, j+1-1), lda)
			}
		}
	} else {
		//        Compute the Cholesky factorization A = L*L**H.
		for j = 1; j <= (*n); j++ {
			//           Compute L(J,J) and test for non-positive-definiteness.
			ajj = a.GetRe(j-1, j-1) - real(goblas.Zdotc(toPtr(j-1), a.CVector(j-1, 0), lda, a.CVector(j-1, 0), lda))
			if ajj <= zero || Disnan(int(ajj)) {
				a.SetRe(j-1, j-1, ajj)
				goto label30
			}
			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J.
			if j < (*n) {
				Zlacgv(toPtr(j-1), a.CVector(j-1, 0), lda)
				goblas.Zgemv(NoTrans, toPtr((*n)-j), toPtr(j-1), toPtrc128(-cone), a.Off(j+1-1, 0), lda, a.CVector(j-1, 0), lda, &cone, a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(j-1), a.CVector(j-1, 0), lda)
				goblas.Zdscal(toPtr((*n)-j), toPtrf64(one/ajj), a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
			}
		}
	}
	return

label30:
	;
	(*info) = j
}
