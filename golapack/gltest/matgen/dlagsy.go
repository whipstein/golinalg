package matgen

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlagsy generates a real symmetric matrix A, by pre- and post-
// multiplying a real diagonal matrix D with a random orthogonal matrix:
// A = U*D*U'. The semi-bandwidth may then be reduced to k by additional
// orthogonal transformations.
func Dlagsy(n *int, k *int, d *mat.Vector, a *mat.Matrix, lda *int, iseed *[]int, work *mat.Vector, info *int) {
	var alpha, half, one, tau, wa, wb, wn, zero float64
	var i, j int

	zero = 0.0
	one = 1.0
	half = 0.5

	//     Test the input arguments
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*k) < 0 || (*k) > (*n)-1 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("DLAGSY"), -(*info))
		return
	}

	//     initialize lower triangle of A to diagonal matrix
	for j = 1; j <= (*n); j++ {
		for i = j + 1; i <= (*n); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= (*n); i++ {
		a.Set(i-1, i-1, d.Get(i-1))
	}

	//     Generate lower triangle of symmetric matrix
	for i = (*n) - 1; i >= 1; i-- {
		//        generate random reflection
		golapack.Dlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*n)-i+1), work)
		wn = goblas.Dnrm2(toPtr((*n)-i+1), work, toPtr(1))
		wa = math.Copysign(wn, work.Get(0))
		if wn == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Dscal(toPtr((*n)-i), toPtrf64(one/wb), work.Off(1), toPtr(1))
			work.Set(0, one)
			tau = wb / wa
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * u
		goblas.Dsymv(mat.Lower, toPtr((*n)-i+1), toPtrf64(tau), a.Off(i-1, i-1), lda, work, toPtr(1), &zero, work.Off((*n)+1-1), toPtr(1))

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Ddot(toPtr((*n)-i+1), work.Off((*n)+1-1), toPtr(1), work, toPtr(1))
		goblas.Daxpy(toPtr((*n)-i+1), &alpha, work, toPtr(1), work.Off((*n)+1-1), toPtr(1))

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		goblas.Dsyr2(mat.Lower, toPtr((*n)-i+1), toPtrf64(-one), work, toPtr(1), work.Off((*n)+1-1), toPtr(1), a.Off(i-1, i-1), lda)
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= (*n)-1-(*k); i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = goblas.Dnrm2(toPtr((*n)-(*k)-i+1), a.Vector((*k)+i-1, i-1), toPtr(1))
		wa = math.Copysign(wn, a.Get((*k)+i-1, i-1))
		if wn == zero {
			tau = zero
		} else {
			wb = a.Get((*k)+i-1, i-1) + wa
			goblas.Dscal(toPtr((*n)-(*k)-i), toPtrf64(one/wb), a.Vector((*k)+i+1-1, i-1), toPtr(1))
			a.Set((*k)+i-1, i-1, one)
			tau = wb / wa
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		goblas.Dgemv(mat.Trans, toPtr((*n)-(*k)-i+1), toPtr((*k)-1), &one, a.Off((*k)+i-1, i+1-1), lda, a.Vector((*k)+i-1, i-1), toPtr(1), &zero, work, toPtr(1))
		goblas.Dger(toPtr((*n)-(*k)-i+1), toPtr((*k)-1), toPtrf64(-tau), a.Vector((*k)+i-1, i-1), toPtr(1), work, toPtr(1), a.Off((*k)+i-1, i+1-1), lda)

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * u
		goblas.Dsymv(mat.Lower, toPtr((*n)-(*k)-i+1), &tau, a.Off((*k)+i-1, (*k)+i-1), lda, a.Vector((*k)+i-1, i-1), toPtr(1), &zero, work, toPtr(1))

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Ddot(toPtr((*n)-(*k)-i+1), work, toPtr(1), a.Vector((*k)+i-1, i-1), toPtr(1))
		goblas.Daxpy(toPtr((*n)-(*k)-i+1), &alpha, a.Vector((*k)+i-1, i-1), toPtr(1), work, toPtr(1))

		//        apply symmetric rank-2 update to A(k+i:n,k+i:n)
		goblas.Dsyr2(mat.Lower, toPtr((*n)-(*k)-i+1), toPtrf64(-one), a.Vector((*k)+i-1, i-1), toPtr(1), work, toPtr(1), a.Off((*k)+i-1, (*k)+i-1), lda)

		a.Set((*k)+i-1, i-1, -wa)
		for j = (*k) + i + 1; j <= (*n); j++ {
			a.Set(j-1, i-1, zero)
		}
	}

	//     Store full symmetric matrix
	for j = 1; j <= (*n); j++ {
		for i = j + 1; i <= (*n); i++ {
			a.Set(j-1, i-1, a.Get(i-1, j-1))
		}
	}
}
