package matgen

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlagsy generates a complex symmetric matrix A, by pre- and post-
// multiplying a real diagonal matrix D with a random unitary matrix:
// A = U*D*U**T. The semi-bandwidth may then be reduced to k by
// additional unitary transformations.
func Zlagsy(n, k int, d *mat.Vector, a *mat.CMatrix, iseed *[]int, work *mat.CVector) (err error) {
	var alpha, half, one, tau, wa, wb, zero complex128
	var wn float64
	var i, ii, j, jj int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Test the input arguments
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 || k > n-1 {
		err = fmt.Errorf("k < 0 || k > n-1: k=%v, n=%v", k, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlagsy", err)
		return
	}

	//     initialize lower triangle of A to diagonal matrix
	for j = 1; j <= n; j++ {
		for i = j + 1; i <= n; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= n; i++ {
		a.SetRe(i-1, i-1, d.Get(i-1))
	}

	//     Generate lower triangle of symmetric matrix
	for i = n - 1; i >= 1; i-- {
		//        generate random reflection
		golapack.Zlarnv(3, iseed, n-i+1, work)
		wn = work.Nrm2(n-i+1, 1)
		wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			work.Off(1).Scal(n-i, one/wb, 1)
			work.Set(0, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * conjg(u)
		golapack.Zlacgv(n-i+1, work, 1)
		if err = golapack.Zsymv(Lower, n-i+1, tau, a.Off(i-1, i-1), work, 1, zero, work.Off(n), 1); err != nil {
			panic(err)
		}
		golapack.Zlacgv(n-i+1, work, 1)

		//        compute  v := y - 1/2 * tau * ( u, y ) * u
		alpha = -half * tau * work.Off(n).Dotc(n-i+1, work, 1, 1)
		work.Off(n).Axpy(n-i+1, alpha, work, 1, 1)

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		//
		//        CALL ZSYR2( 'Lower', N-I+1, -ONE, WORK, 1, WORK( N+1 ), 1,
		//        $               A( I, I ), LDA )
		for jj = i; jj <= n; jj++ {
			for ii = jj; ii <= n; ii++ {
				a.Set(ii-1, jj-1, a.Get(ii-1, jj-1)-work.Get(ii-i)*work.Get(n+jj-i)-work.Get(n+ii-i)*work.Get(jj-i))
			}
		}
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= n-1-k; i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = a.Off(k+i-1, i-1).CVector().Nrm2(n-k-i+1, 1)
		wa = complex(wn/a.GetMag(k+i-1, i-1), 0) * a.Get(k+i-1, i-1)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = a.Get(k+i-1, i-1) + wa
			a.Off(k+i, i-1).CVector().Scal(n-k-i, one/wb, 1)
			a.Set(k+i-1, i-1, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		if err = work.Gemv(ConjTrans, n-k-i+1, k-1, one, a.Off(k+i-1, i), a.Off(k+i-1, i-1).CVector(), 1, zero, 1); err != nil {
			panic(err)
		}
		if err = a.Off(k+i-1, i).Gerc(n-k-i+1, k-1, -tau, a.Off(k+i-1, i-1).CVector(), 1, work, 1); err != nil {
			panic(err)
		}

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * conjg(u)
		golapack.Zlacgv(n-k-i+1, a.Off(k+i-1, i-1).CVector(), 1)
		if err = golapack.Zsymv(Lower, n-k-i+1, tau, a.Off(k+i-1, k+i-1), a.Off(k+i-1, i-1).CVector(), 1, zero, work, 1); err != nil {
			panic(err)
		}
		golapack.Zlacgv(n-k-i+1, a.Off(k+i-1, i-1).CVector(), 1)

		//        compute  v := y - 1/2 * tau * ( u, y ) * u
		alpha = -half * tau * work.Dotc(n-k-i+1, a.Off(k+i-1, i-1).CVector(), 1, 1)
		work.Axpy(n-k-i+1, alpha, a.Off(k+i-1, i-1).CVector(), 1, 1)

		//        apply symmetric rank-2 update to A(k+i:n,k+i:n)
		//
		//        CALL ZSYR2( 'Lower', N-K-I+1, -ONE, A( K+I, I ), 1, WORK, 1,
		//        $               A( K+I, K+I ), LDA )
		for jj = k + i; jj <= n; jj++ {
			for ii = jj; ii <= n; ii++ {
				a.Set(ii-1, jj-1, a.Get(ii-1, jj-1)-a.Get(ii-1, i-1)*work.Get(jj-k-i)-work.Get(ii-k-i)*a.Get(jj-1, i-1))
			}
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
