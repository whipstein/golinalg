package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zptcon computes the reciprocal of the condition number (in the
// 1-norm) of a complex Hermitian positive definite tridiagonal matrix
// using the factorization A = L*D*L**H or A = U**H*D*U computed by
// ZPTTRF.
//
// Norm(inv(A)) is computed by a direct method, and the reciprocal of
// the condition number is computed as
//                  RCOND = 1 / (ANORM * norm(inv(A))).
func Zptcon(n int, d *mat.Vector, e *mat.CVector, anorm float64, rwork *mat.Vector) (rcond float64, err error) {
	var ainvnm, one, zero float64
	var i, ix int

	one = 1.0
	zero = 0.0

	//     Test the input arguments.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if anorm < zero {
		err = fmt.Errorf("anorm < zero: anorm=%v", anorm)
	}
	if err != nil {
		gltest.Xerbla2("Zptcon", err)
		return
	}

	//     Quick return if possible
	rcond = zero
	if n == 0 {
		rcond = one
		return
	} else if anorm == zero {
		return
	}

	//     Check that D(1:N) is positive.
	for i = 1; i <= n; i++ {
		if d.Get(i-1) <= zero {
			return
		}
	}

	//     Solve M(A) * x = e, where M(A) = (m(i,j)) is given by
	//
	//        m(i,j) =  abs(A(i,j)), i = j,
	//        m(i,j) = -abs(A(i,j)), i .ne. j,
	//
	//     and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**H.
	//
	//     Solve M(L) * x = e.
	rwork.Set(0, one)
	for i = 2; i <= n; i++ {
		rwork.Set(i-1, one+rwork.Get(i-1-1)*e.GetMag(i-1-1))
	}

	//     Solve D * M(L)**H * x = b.
	rwork.Set(n-1, rwork.Get(n-1)/d.Get(n-1))
	for i = n - 1; i >= 1; i-- {
		rwork.Set(i-1, rwork.Get(i-1)/d.Get(i-1)+rwork.Get(i)*e.GetMag(i-1))
	}

	//     Compute AINVNM = max(x(i)), 1<=i<=n.
	ix = rwork.Iamax(n, 1)
	ainvnm = rwork.GetMag(ix - 1)

	//     Compute the reciprocal condition number.
	if ainvnm != zero {
		rcond = (one / ainvnm) / anorm
	}

	return
}
