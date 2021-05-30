package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dspcon estimates the reciprocal of the condition number (in the
// 1-norm) of a real symmetric packed matrix A using the factorization
// A = U*D*U**T or A = L*D*L**T computed by DSPTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Dspcon(uplo byte, n *int, ap *mat.Vector, ipiv *[]int, anorm, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var upper bool
	var ainvnm, one, zero float64
	var i, ip, kase int
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
	} else if (*anorm) < zero {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPCON"), -(*info))
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
		ip = (*n) * ((*n) + 1) / 2
		for i = (*n); i >= 1; i-- {
			if (*ipiv)[i-1] > 0 && ap.Get(ip-1) == zero {
				return
			}
			ip = ip - i
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		ip = 1
		for i = 1; i <= (*n); i++ {
			if (*ipiv)[i-1] > 0 && ap.Get(ip-1) == zero {
				return
			}
			ip = ip + (*n) - i + 1
		}
	}

	//     Estimate the 1-norm of the inverse.
	kase = 0
label30:
	;
	Dlacn2(n, work.Off((*n)+1-1), work, iwork, &ainvnm, &kase, &isave)
	if kase != 0 {
		//        Multiply by inv(L*D*L**T) or inv(U*D*U**T).
		Dsptrs(uplo, n, func() *int { y := 1; return &y }(), ap, ipiv, work.Matrix(*n, opts), n, info)
		goto label30
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
