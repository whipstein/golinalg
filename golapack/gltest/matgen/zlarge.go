package matgen

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlarge pre- and post-multiplies a complex general n by n matrix A
// with a random unitary matrix: A = U*D*U'.
func Zlarge(n *int, a *mat.CMatrix, lda *int, iseed *[]int, work *mat.CVector, info *int) {
	var one, tau, wa, wb, zero complex128
	var wn float64
	var i int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*lda) < maxint(1, *n) {
		(*info) = -3
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("ZLARGE"), -(*info))
		return
	}

	//     pre- and post-multiply A by random unitary matrix
	for i = (*n); i >= 1; i-- { //
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

		//        multiply A(i:n,1:n) by random reflection from the left
		goblas.Zgemv(ConjTrans, toPtr((*n)-i+1), n, &one, a.Off(i-1, 0), lda, work, func() *int { y := 1; return &y }(), &zero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
		goblas.Zgerc(toPtr((*n)-i+1), n, toPtrc128(-tau), work, func() *int { y := 1; return &y }(), work.Off((*n)+1-1), func() *int { y := 1; return &y }(), a.Off(i-1, 0), lda)

		//        multiply A(1:n,i:n) by random reflection from the right
		goblas.Zgemv(NoTrans, n, toPtr((*n)-i+1), &one, a.Off(0, i-1), lda, work, func() *int { y := 1; return &y }(), &zero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
		goblas.Zgerc(n, toPtr((*n)-i+1), toPtrc128(-tau), work.Off((*n)+1-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }(), a.Off(0, i-1), lda)
	}
}
