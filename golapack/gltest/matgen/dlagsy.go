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
	var err error
	_ = err

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
		wn = goblas.Dnrm2((*n)-i+1, work, 1)
		wa = math.Copysign(wn, work.Get(0))
		if wn == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Dscal((*n)-i, one/wb, work.Off(1), 1)
			work.Set(0, one)
			tau = wb / wa
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * u
		err = goblas.Dsymv(mat.Lower, (*n)-i+1, tau, a.Off(i-1, i-1), *lda, work, 1, zero, work.Off((*n)+1-1), 1)

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Ddot((*n)-i+1, work.Off((*n)+1-1), 1, work, 1)
		goblas.Daxpy((*n)-i+1, alpha, work, 1, work.Off((*n)+1-1), 1)

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		err = goblas.Dsyr2(mat.Lower, (*n)-i+1, -one, work, 1, work.Off((*n)+1-1), 1, a.Off(i-1, i-1), *lda)
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= (*n)-1-(*k); i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = goblas.Dnrm2((*n)-(*k)-i+1, a.Vector((*k)+i-1, i-1), 1)
		wa = math.Copysign(wn, a.Get((*k)+i-1, i-1))
		if wn == zero {
			tau = zero
		} else {
			wb = a.Get((*k)+i-1, i-1) + wa
			goblas.Dscal((*n)-(*k)-i, one/wb, a.Vector((*k)+i+1-1, i-1), 1)
			a.Set((*k)+i-1, i-1, one)
			tau = wb / wa
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		err = goblas.Dgemv(mat.Trans, (*n)-(*k)-i+1, (*k)-1, one, a.Off((*k)+i-1, i+1-1), *lda, a.Vector((*k)+i-1, i-1), 1, zero, work, 1)
		err = goblas.Dger((*n)-(*k)-i+1, (*k)-1, -tau, a.Vector((*k)+i-1, i-1), 1, work, 1, a.Off((*k)+i-1, i+1-1), *lda)

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * u
		err = goblas.Dsymv(mat.Lower, (*n)-(*k)-i+1, tau, a.Off((*k)+i-1, (*k)+i-1), *lda, a.Vector((*k)+i-1, i-1), 1, zero, work, 1)

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Ddot((*n)-(*k)-i+1, work, 1, a.Vector((*k)+i-1, i-1), 1)
		goblas.Daxpy((*n)-(*k)-i+1, alpha, a.Vector((*k)+i-1, i-1), 1, work, 1)

		//        apply symmetric rank-2 update to A(k+i:n,k+i:n)
		err = goblas.Dsyr2(mat.Lower, (*n)-(*k)-i+1, -one, a.Vector((*k)+i-1, i-1), 1, work, 1, a.Off((*k)+i-1, (*k)+i-1), *lda)

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
