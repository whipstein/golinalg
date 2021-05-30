package matgen

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlagge generates a complex general m by n matrix A, by pre- and post-
// multiplying a real diagonal matrix D with random unitary matrices:
// A = U*D*V. The lower and upper bandwidths may then be reduced to
// kl and ku by additional unitary transformations.
func Zlagge(m, n, kl, ku *int, d *mat.Vector, a *mat.CMatrix, lda *int, iseed *[]int, work *mat.CVector, info *int) {
	var one, tau, wa, wb, zero complex128
	var wn float64
	var i, j int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kl) < 0 || (*kl) > (*m)-1 {
		(*info) = -3
	} else if (*ku) < 0 || (*ku) > (*n)-1 {
		(*info) = -4
	} else if (*lda) < maxint(1, *m) {
		(*info) = -7
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("ZLAGGE"), -(*info))
		return
	}

	//     initialize A to diagonal matrix
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= minint(*m, *n); i++ {
		a.Set(i-1, i-1, complex(d.Get(i-1), 0))
	}

	//     Quick exit if the user wants a diagonal matrix
	if ((*kl) == 0) && ((*ku) == 0) {
		return
	}

	//     pre- and post-multiply A by random unitary matrices
	for i = minint(*m, *n); i >= 1; i-- {
		if i < (*m) {
			//           generate random reflection
			golapack.Zlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*m)-i+1), work)
			wn = goblas.Dznrm2(toPtr((*m)-i+1), work, func() *int { y := 1; return &y }())
			wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
			if complex(wn, 0) == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				goblas.Zscal(toPtr((*m)-i), toPtrc128(one/wb), work.Off(1), func() *int { y := 1; return &y }())
				work.Set(0, one)
				tau = wb / wa
			}

			//           multiply A(i:m,i:n) by random reflection from the left
			goblas.Zgemv(ConjTrans, toPtr((*m)-i+1), toPtr((*n)-i+1), &one, a.Off(i-1, i-1), lda, work, func() *int { y := 1; return &y }(), &zero, work.Off((*m)+1-1), func() *int { y := 1; return &y }())
			goblas.Zgerc(toPtr((*m)-i+1), toPtr((*n)-i+1), toPtrc128(-tau), work, func() *int { y := 1; return &y }(), work.Off((*m)+1-1), func() *int { y := 1; return &y }(), a.Off(i-1, i-1), lda)
		}
		if i < (*n) {
			//           generate random reflection
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

			//           multiply A(i:m,i:n) by random reflection from the right
			goblas.Zgemv(NoTrans, toPtr((*m)-i+1), toPtr((*n)-i+1), &one, a.Off(i-1, i-1), lda, work, func() *int { y := 1; return &y }(), &zero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
			goblas.Zgerc(toPtr((*m)-i+1), toPtr((*n)-i+1), toPtrc128(-tau), work.Off((*n)+1-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }(), a.Off(i-1, i-1), lda)
		}
	}

	//     Reduce number of subdiagonals to KL and number of superdiagonals
	//     to KU
	for i = 1; i <= maxint((*m)-1-(*kl), (*n)-1-(*ku)); i++ {
		if (*kl) <= (*ku) {
			//           annihilate subdiagonal elements first (necessary if KL = 0)
			if i <= minint((*m)-1-(*kl), *n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = goblas.Dznrm2(toPtr((*m)-(*kl)-i+1), a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }())
				wa = complex(wn/a.GetMag((*kl)+i-1, i-1), 0) * a.Get((*kl)+i-1, i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get((*kl)+i-1, i-1) + wa
					goblas.Zscal(toPtr((*m)-(*kl)-i), toPtrc128(one/wb), a.CVector((*kl)+i+1-1, i-1), func() *int { y := 1; return &y }())
					a.Set((*kl)+i-1, i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				goblas.Zgemv(ConjTrans, toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), &one, a.Off((*kl)+i-1, i+1-1), lda, a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				goblas.Zgerc(toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), toPtrc128(-tau), a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }(), a.Off((*kl)+i-1, i+1-1), lda)
				a.Set((*kl)+i-1, i-1, -wa)
			}

			if i <= minint((*n)-1-(*ku), *m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dznrm2(toPtr((*n)-(*ku)-i+1), a.CVector(i-1, (*ku)+i-1), lda)
				wa = complex(wn/a.GetMag(i-1, (*ku)+i-1), 0) * a.Get(i-1, (*ku)+i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, (*ku)+i-1) + wa
					goblas.Zscal(toPtr((*n)-(*ku)-i), toPtrc128(one/wb), a.CVector(i-1, (*ku)+i+1-1), lda)
					a.Set(i-1, (*ku)+i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				golapack.Zlacgv(toPtr((*n)-(*ku)-i+1), a.CVector(i-1, (*ku)+i-1), lda)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), &one, a.Off(i+1-1, (*ku)+i-1), lda, a.CVector(i-1, (*ku)+i-1), lda, &zero, work, func() *int { y := 1; return &y }())
				goblas.Zgerc(toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), toPtrc128(-tau), work, func() *int { y := 1; return &y }(), a.CVector(i-1, (*ku)+i-1), lda, a.Off(i+1-1, (*ku)+i-1), lda)
				a.Set(i-1, (*ku)+i-1, -wa)
			}
		} else {
			//           annihilate superdiagonal elements first (necessary if
			//           KU = 0)
			if i <= minint((*n)-1-(*ku), *m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dznrm2(toPtr((*n)-(*ku)-i+1), a.CVector(i-1, (*ku)+i-1), lda)
				wa = complex(wn/a.GetMag(i-1, (*ku)+i-1), 0) * a.Get(i-1, (*ku)+i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, (*ku)+i-1) + wa
					goblas.Zscal(toPtr((*n)-(*ku)-i), toPtrc128(one/wb), a.CVector(i-1, (*ku)+i+1-1), lda)
					a.Set(i-1, (*ku)+i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				golapack.Zlacgv(toPtr((*n)-(*ku)-i+1), a.CVector(i-1, (*ku)+i-1), lda)
				goblas.Zgemv(NoTrans, toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), &one, a.Off(i+1-1, (*ku)+i-1), lda, a.CVector(i-1, (*ku)+i-1), lda, &zero, work, func() *int { y := 1; return &y }())
				goblas.Zgerc(toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), toPtrc128(-tau), work, func() *int { y := 1; return &y }(), a.CVector(i-1, (*ku)+i-1), lda, a.Off(i+1-1, (*ku)+i-1), lda)
				a.Set(i-1, (*ku)+i-1, -wa)
			}

			if i <= minint((*m)-1-(*kl), *n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = goblas.Dznrm2(toPtr((*m)-(*kl)-i+1), a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }())
				wa = complex(wn/a.GetMag((*kl)+i-1, i-1), 0) * a.Get((*kl)+i-1, i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get((*kl)+i-1, i-1) + wa
					goblas.Zscal(toPtr((*m)-(*kl)-i), toPtrc128(one/wb), a.CVector((*kl)+i+1-1, i-1), func() *int { y := 1; return &y }())
					a.Set((*kl)+i-1, i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				goblas.Zgemv(ConjTrans, toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), &one, a.Off((*kl)+i-1, i+1-1), lda, a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				goblas.Zgerc(toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), toPtrc128(-tau), a.CVector((*kl)+i-1, i-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }(), a.Off((*kl)+i-1, i+1-1), lda)
				a.Set((*kl)+i-1, i-1, -wa)
			}
		}

		if i <= (*n) {
			for j = (*kl) + i + 1; j <= (*m); j++ {
				a.Set(j-1, i-1, zero)
			}
		}

		if i <= (*m) {
			for j = (*ku) + i + 1; j <= (*n); j++ {
				a.Set(i-1, j-1, zero)
			}
		}
	}
}