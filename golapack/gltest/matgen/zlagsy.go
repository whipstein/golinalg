package matgen

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlagsy generates a complex symmetric matrix A, by pre- and post-
// multiplying a real diagonal matrix D with a random unitary matrix:
// A = U*D*U**T. The semi-bandwidth may then be reduced to k by
// additional unitary transformations.
func Zlagsy(n, k *int, d *mat.Vector, a *mat.CMatrix, lda *int, iseed *[]int, work *mat.CVector, info *int) {
	var alpha, half, one, tau, wa, wb, zero complex128
	var wn float64
	var i, ii, j, jj int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZLAGSY"), -(*info))
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

	//     Generate lower triangle of symmetric matrix
	for i = (*n) - 1; i >= 1; i-- {
		//        generate random reflection
		golapack.Zlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*n)-i+1), work)
		wn = goblas.Dznrm2(toPtr((*n)-i+1), work, func() *int { y := 1; return &y }())
		wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Zscal(toPtr((*n)-i), toPtrc128(one/wb), work.Off(1), func() *int { y := 1; return &y }())
			work.Set(0, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply random reflection to A(i:n,i:n) from the left
		//        and the right
		//
		//        compute  y := tau * A * conjg(u)
		golapack.Zlacgv(toPtr((*n)-i+1), work, func() *int { y := 1; return &y }())
		golapack.Zsymv('L', toPtr((*n)-i+1), &tau, a.Off(i-1, i-1), lda, work, func() *int { y := 1; return &y }(), &zero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
		golapack.Zlacgv(toPtr((*n)-i+1), work, func() *int { y := 1; return &y }())

		//        compute  v := y - 1/2 * tau * ( u, y ) * u
		alpha = -half * tau * goblas.Zdotc(toPtr((*n)-i+1), work, func() *int { y := 1; return &y }(), work.Off((*n)+1-1), func() *int { y := 1; return &y }())
		goblas.Zaxpy(toPtr((*n)-i+1), &alpha, work, func() *int { y := 1; return &y }(), work.Off((*n)+1-1), func() *int { y := 1; return &y }())

		//        apply the transformation as a rank-2 update to A(i:n,i:n)
		//
		//        CALL ZSYR2( 'Lower', N-I+1, -ONE, WORK, 1, WORK( N+1 ), 1,
		//        $               A( I, I ), LDA )
		for jj = i; jj <= (*n); jj++ {
			for ii = jj; ii <= (*n); ii++ {
				a.Set(ii-1, jj-1, a.Get(ii-1, jj-1)-work.Get(ii-i+1-1)*work.Get((*n)+jj-i+1-1)-work.Get((*n)+ii-i+1-1)*work.Get(jj-i+1-1))
			}
		}
	}

	//     Reduce number of subdiagonals to K
	for i = 1; i <= (*n)-1-(*k); i++ {
		//        generate reflection to annihilate A(k+i+1:n,i)
		wn = goblas.Dznrm2(toPtr((*n)-(*k)-i+1), a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }())
		wa = complex(wn/a.GetMag((*k)+i-1, i-1), 0) * a.Get((*k)+i-1, i-1)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = a.Get((*k)+i-1, i-1) + wa
			goblas.Zscal(toPtr((*n)-(*k)-i), toPtrc128(one/wb), a.CVector((*k)+i+1-1, i-1), func() *int { y := 1; return &y }())
			a.Set((*k)+i-1, i-1, one)
			tau = complex(real(wb/wa), 0)
		}

		//        apply reflection to A(k+i:n,i+1:k+i-1) from the left
		goblas.Zgemv(ConjTrans, toPtr((*n)-(*k)-i+1), toPtr((*k)-1), &one, a.Off((*k)+i-1, i+1-1), lda, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
		goblas.Zgerc(toPtr((*n)-(*k)-i+1), toPtr((*k)-1), toPtrc128(-tau), a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }(), a.Off((*k)+i-1, i+1-1), lda)

		//        apply reflection to A(k+i:n,k+i:n) from the left and the right
		//
		//        compute  y := tau * A * conjg(u)
		golapack.Zlacgv(toPtr((*n)-(*k)-i+1), a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }())
		golapack.Zsymv('L', toPtr((*n)-(*k)-i+1), &tau, a.Off((*k)+i-1, (*k)+i-1), lda, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
		golapack.Zlacgv(toPtr((*n)-(*k)-i+1), a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }())

		//        compute  v := y - 1/2 * tau * ( u, y ) * u
		alpha = -half * tau * goblas.Zdotc(toPtr((*n)-(*k)-i+1), a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		goblas.Zaxpy(toPtr((*n)-(*k)-i+1), &alpha, a.CVector((*k)+i-1, i-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())

		//        apply symmetric rank-2 update to A(k+i:n,k+i:n)
		//
		//        CALL ZSYR2( 'Lower', N-K-I+1, -ONE, A( K+I, I ), 1, WORK, 1,
		//        $               A( K+I, K+I ), LDA )
		for jj = (*k) + i; jj <= (*n); jj++ {
			for ii = jj; ii <= (*n); ii++ {
				a.Set(ii-1, jj-1, a.Get(ii-1, jj-1)-a.Get(ii-1, i-1)*work.Get(jj-(*k)-i+1-1)-work.Get(ii-(*k)-i+1-1)*a.Get(jj-1, i-1))
			}
		}

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
