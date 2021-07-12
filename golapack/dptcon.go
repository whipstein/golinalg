package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dptcon computes the reciprocal of the condition number (in the
// 1-norm) of a real symmetric positive definite tridiagonal matrix
// using the factorization A = L*D*L**T or A = U**T*D*U computed by
// DPTTRF.
//
// Norm(inv(A)) is computed by a direct method, and the reciprocal of
// the condition number is computed as
//              RCOND = 1 / (ANORM * norm(inv(A))).
func Dptcon(n *int, d, e *mat.Vector, anorm *float64, rcond *float64, work *mat.Vector, info *int) {
	var ainvnm, one, zero float64
	var i, ix int

	one = 1.0
	zero = 0.0

	//     Test the input arguments.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*anorm) < zero {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPTCON"), -(*info))
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

	//     Check that D(1:N) is positive.
	for i = 1; i <= (*n); i++ {
		if d.Get(i-1) <= zero {
			return
		}
	}

	//     Solve M(A) * x = e, where M(A) = (m(i,j)) is given by
	//
	//        m(i,j) =  abs(A(i,j)), i = j,
	//        m(i,j) = -abs(A(i,j)), i .ne. j,
	//
	//     and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**T.
	//
	//     Solve M(L) * x = e.
	work.Set(0, one)
	for i = 2; i <= (*n); i++ {
		work.Set(i-1, one+work.Get(i-1-1)*math.Abs(e.Get(i-1-1)))
	}

	//     Solve D * M(L)**T * x = b.
	work.Set((*n)-1, work.Get((*n)-1)/d.Get((*n)-1))
	for i = (*n) - 1; i >= 1; i-- {
		work.Set(i-1, work.Get(i-1)/d.Get(i-1)+work.Get(i)*math.Abs(e.Get(i-1)))
	}

	//     Compute AINVNM = max(x(i)), 1<=i<=n.
	ix = goblas.Idamax(*n, work.Off(0, 1))
	ainvnm = math.Abs(work.Get(ix - 1))

	//     Compute the reciprocal condition number.
	if ainvnm != zero {
		(*rcond) = (one / ainvnm) / (*anorm)
	}
}
