package matgen

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarge pre- and post-multiplies a real general n by n matrix A
// with a random orthogonal matrix: A = U*D*U'.
func Dlarge(n *int, a *mat.Matrix, lda *int, iseed *[]int, work *mat.Vector, info *int) {
	var one, tau, wa, wb, wn, zero float64
	var i int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*lda) < max(1, *n) {
		(*info) = -3
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("DLARGE"), -(*info))
		return
	}

	//     pre- and post-multiply A by random orthogonal matrix
	for i = (*n); i >= 1; i-- {
		//        generate random reflection
		golapack.Dlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*n)-i+1), work)
		wn = goblas.Dnrm2((*n)-i+1, work.Off(0, 1))
		wa = math.Copysign(wn, work.Get(0))
		if wn == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Dscal((*n)-i, one/wb, work.Off(1, 1))
			work.Set(0, one)
			tau = wb / wa
		}

		//        multiply A(i:n,1:n) by random reflection from the left
		err = goblas.Dgemv(Trans, (*n)-i+1, *n, one, a.Off(i-1, 0), work.Off(0, 1), zero, work.Off((*n), 1))
		err = goblas.Dger((*n)-i+1, *n, -tau, work.Off(0, 1), work.Off((*n), 1), a.Off(i-1, 0))

		//        multiply A(1:n,i:n) by random reflection from the right
		err = goblas.Dgemv(NoTrans, *n, (*n)-i+1, one, a.Off(0, i-1), work.Off(0, 1), zero, work.Off((*n), 1))
		err = goblas.Dger(*n, (*n)-i+1, -tau, work.Off((*n), 1), work.Off(0, 1), a.Off(0, i-1))
	}
}
