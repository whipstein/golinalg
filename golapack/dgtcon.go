package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgtcon estimates the reciprocal of the condition number of a real
// tridiagonal matrix A using the LU factorization as computed by
// DGTTRF.
//
// An estimate is obtained for norm(inv(A)), and the reciprocal of the
// condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
func Dgtcon(norm byte, n *int, dl, d, du, du2 *mat.Vector, ipiv *[]int, anorm, rcond *float64, work *mat.Vector, iwork *[]int, info *int) {
	var onenrm bool
	var ainvnm, one, zero float64
	var i, kase, kase1 int
	isave := make([]int, 3)

	one = 1.0
	zero = 0.0

	//     Test the input arguments.
	(*info) = 0
	onenrm = norm == '1' || norm == 'O'
	if !onenrm && norm != 'I' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*anorm) < zero {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGTCON"), -(*info))
		return
	}

	//     Quick return if possible
	(*rcond) = zero
	if (*n) == 0 {
		(*rcond) = one
		return
	} else if (*anorm) == zero {
		return
	}

	//     Check that D(1:N) is non-zero.
	for i = 1; i <= (*n); i++ {
		if d.Get(i-1) == zero {
			return
		}
	}

	ainvnm = zero
	if onenrm {
		kase1 = 1
	} else {
		kase1 = 2
	}
	kase = 0
label20:
	;
	Dlacn2(n, work.Off((*n)+1-1), work.Off(0), iwork, &ainvnm, &kase, &isave)
	if kase != 0 {
		if kase == kase1 {
			//           Multiply by inv(U)*inv(L).
			Dgttrs('N', n, func() *int { y := 1; return &y }(), dl, d, du, du2, ipiv, work.Matrix(*n, opts), n, info)
		} else {
			//           Multiply by inv(L**T)*inv(U**T).
			Dgttrs('T', n, func() *int { y := 1; return &y }(), dl, d, du, du2, ipiv, work.Matrix(*n, opts), n, info)
		}
		goto label20
	}

	//     Compute the estimate of the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
