package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsycon estimates the reciprocal of the condition number (in the
// 1-norm) of a real symmetric matrix A using the factorization
// A = U*D*U**T or A = L*D*L**T computed by DSYTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Dsycon(uplo byte, n *int, a *mat.Matrix, lda *int, ipiv *[]int, anorm *float64, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var upper bool
	var ainvnm, one, zero float64
	var i, kase int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	} else if (*anorm) < zero {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYCON"), -(*info))
		return
	}

	//     Quick return if possible
	(*rcond) = zero
	if (*n) == 0 {
		(*rcond) = one
		return
	} else if (*anorm) <= zero {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for i = (*n); i >= 1; i-- {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for i = 1; i <= (*n); i++ {
			if (*ipiv)[i-1] > 0 && a.Get(i-1, i-1) == zero {
				return
			}
		}
	}

	//     Estimate the 1-norm of the inverse.
	kase = 0
label30:
	;
	Dlacn2(n, work.Off((*n)), work, iwork, &ainvnm, &kase, &isave)
	if kase != 0 {
		//        Multiply by inv(L*D*L**T) or inv(U*D*U**T).
		Dsytrs(uplo, n, func() *int { y := 1; return &y }(), a, lda, ipiv, work.Matrix(*n, opts), n, info)
		goto label30
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
