package matgen

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlaghe generates a complex hermitian matrix A, by pre- and post-
// multiplying a real diagonal matrix D with a random unitary matrix:
// A = U*D*U'. The semi-bandwidth may then be reduced to k by additional
// unitary transformations.
func Zlaghe(n, k *int, d *mat.Vector, a *mat.CMatrix, lda *int, iseed *[]int, work *mat.CVector, info *int) {
	var alpha, half, one, tau, wa, wb, zero complex128
	var wn float64
	var i, j int
	var err error
	_ = err

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*k) < 0 || (*k) > (*n)-1 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("ZLAGHE"), -(*info))
		return
	}

	//     initialize lower triangle of A to diagonal matrix
	for j = 1; j <= (*n); j++ {
		for i = j + 1; i <= (*n); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= (*n); i++ {
		a.SetRe(i-1, i-1, d.Get(i-1))
	}

	//     Generate lower triangle of hermitian matrix
	for i = (*n) - 1; i >= 1; i-- {
		//        generate random reflection
		golapack.Zlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*n)-i+1), work)
		wn = goblas.Dznrm2((*n)-i+1, work.Off(0, 1))
		wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Zscal((*n)-i, one/wb, work.Off(1, 1))
			work.Set(0, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * u
		err = goblas.Zhemv(Lower, (*n)-i+1, tau, a.Off(i-1, i-1), work.Off(0, 1), zero, work.Off((*n), 1))

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Zdotc((*n)-i+1, work.Off((*n), 1), work.Off(0, 1))
		goblas.Zaxpy((*n)-i+1, alpha, work.Off(0, 1), work.Off((*n), 1))

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		err = goblas.Zher2(Lower, (*n)-i+1, -one, work.Off(0, 1), work.Off((*n), 1), a.Off(i-1, i-1))
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= (*n)-1-(*k); i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = goblas.Dznrm2((*n)-(*k)-i+1, a.CVector((*k)+i-1, i-1, 1))
		wa = complex(wn/a.GetMag((*k)+i-1, i-1), 0) * a.Get((*k)+i-1, i-1)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = a.Get((*k)+i-1, i-1) + wa
			goblas.Zscal((*n)-(*k)-i, one/wb, a.CVector((*k)+i, i-1, 1))
			a.Set((*k)+i-1, i-1, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		err = goblas.Zgemv(ConjTrans, (*n)-(*k)-i+1, (*k)-1, one, a.Off((*k)+i-1, i), a.CVector((*k)+i-1, i-1, 1), zero, work.Off(0, 1))
		err = goblas.Zgerc((*n)-(*k)-i+1, (*k)-1, -tau, a.CVector((*k)+i-1, i-1, 1), work.Off(0, 1), a.Off((*k)+i-1, i))

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * u
		err = goblas.Zhemv(Lower, (*n)-(*k)-i+1, tau, a.Off((*k)+i-1, (*k)+i-1), a.CVector((*k)+i-1, i-1, 1), zero, work.Off(0, 1))

		//        compute  v := y - 1/2 * tau * ( y, u ) * u
		alpha = -half * tau * goblas.Zdotc((*n)-(*k)-i+1, work.Off(0, 1), a.CVector((*k)+i-1, i-1, 1))
		goblas.Zaxpy((*n)-(*k)-i+1, alpha, a.CVector((*k)+i-1, i-1, 1), work.Off(0, 1))

		//        apply hermitian rank-2 update to A(k+i:n,k+i:n)
		err = goblas.Zher2(Lower, (*n)-(*k)-i+1, -one, a.CVector((*k)+i-1, i-1, 1), work.Off(0, 1), a.Off((*k)+i-1, (*k)+i-1))

		a.Set((*k)+i-1, i-1, -wa)
		for j = (*k) + i + 1; j <= (*n); j++ {
			a.Set(j-1, i-1, zero)
		}
	}

	//     Store full hermitian matrix
	for j = 1; j <= (*n); j++ {
		for i = j + 1; i <= (*n); i++ {
			a.Set(j-1, i-1, a.GetConj(i-1, j-1))
		}
	}
}
