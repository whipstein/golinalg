package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlagsy generates a real symmetric matrix A, by pre- and post-
// multiplying a real diagonal matrix D with a random orthogonal matrix:
// A = U*D*U'. The semi-bandwidth may then be reduced to k by additional
// orthogonal transformations.
func Dlagsy(n, k int, d *mat.Vector, a *mat.Matrix, iseed *[]int, work *mat.Vector) (err error) {
	var alpha, half, one, tau, wa, wb, wn, zero float64
	var i, j int

	zero = 0.0
	one = 1.0
	half = 0.5

	//     Test the input arguments
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > n-1 {
		err = fmt.Errorf("k < 0 || k > n-1: k=%v", k)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("DLAGSY", err)
		return
	}

	//     initialize lower triangle of A to diagonal matrix
	for j = 1; j <= n; j++ {
		for i = j + 1; i <= n; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= n; i++ {
		a.Set(i-1, i-1, d.Get(i-1))
	}

	//     Generate lower triangle of symmetric matrix
	for i = n - 1; i >= 1; i-- {
		//        generate random reflection
		golapack.Dlarnv(3, iseed, n-i+1, work)
		wn = work.Nrm2(n-i+1, 1)
		wa = math.Copysign(wn, work.Get(0))
		if wn == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			work.Off(1).Scal(n-i, one/wb, 1)
			work.Set(0, one)
			tau = wb / wa
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * u
		if err = work.Off(n).Symv(Lower, n-i+1, tau, a.Off(i-1, i-1), work, 1, zero, 1); err != nil {
			panic(err)
		}

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * work.Dot(n-i+1, work.Off(n), 1, 1)
		work.Off(n).Axpy(n-i+1, alpha, work, 1, 1)

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		if err = a.Off(i-1, i-1).Syr2(Lower, n-i+1, -one, work, 1, work.Off(n), 1); err != nil {
			panic(err)
		}
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= n-1-k; i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = a.Off(k+i-1, i-1).Vector().Nrm2(n-k-i+1, 1)
		wa = math.Copysign(wn, a.Get(k+i-1, i-1))
		if wn == zero {
			tau = zero
		} else {
			wb = a.Get(k+i-1, i-1) + wa
			a.Off(k+i, i-1).Vector().Scal(n-k-i, one/wb, 1)
			a.Set(k+i-1, i-1, one)
			tau = wb / wa
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		if err = work.Gemv(Trans, n-k-i+1, k-1, one, a.Off(k+i-1, i), a.Off(k+i-1, i-1).Vector(), 1, zero, 1); err != nil {
			panic(err)
		}
		if err = a.Off(k+i-1, i).Ger(n-k-i+1, k-1, -tau, a.Off(k+i-1, i-1).Vector(), 1, work, 1); err != nil {
			panic(err)
		}

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * u
		if err = work.Symv(Lower, n-k-i+1, tau, a.Off(k+i-1, k+i-1), a.Off(k+i-1, i-1).Vector(), 1, zero, 1); err != nil {
			panic(err)
		}

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * a.Off(k+i-1, i-1).Vector().Dot(n-k-i+1, work, 1, 1)
		work.Axpy(n-k-i+1, alpha, a.Off(k+i-1, i-1).Vector(), 1, 1)

		//        apply symmetric rank-2 update to A(k+i:n,k+i:n)
		if err = a.Off(k+i-1, k+i-1).Syr2(Lower, n-k-i+1, -one, a.Off(k+i-1, i-1).Vector(), 1, work, 1); err != nil {
			panic(err)
		}

		a.Set(k+i-1, i-1, -wa)
		for j = k + i + 1; j <= n; j++ {
			a.Set(j-1, i-1, zero)
		}
	}

	//     Store full symmetric matrix
	for j = 1; j <= n; j++ {
		for i = j + 1; i <= n; i++ {
			a.Set(j-1, i-1, a.Get(i-1, j-1))
		}
	}

	return
}
