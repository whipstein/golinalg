package matgen

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlagge generates a real general m by n matrix A, by pre- and post-
// multiplying a real diagonal matrix D with random orthogonal matrices:
// A = U*D*V. The lower and upper bandwidths may then be reduced to
// kl and ku by additional orthogonal transformations.
func Dlagge(m *int, n *int, kl *int, ku *int, d *mat.Vector, a *mat.Matrix, lda *int, iseed *[]int, work *mat.Vector, info *int) {
	var one, tau, wa, wb, wn, zero float64
	var i, j int

	zero = 0.0
	one = 1.0

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
		gltest.Xerbla([]byte("DLAGGE"), -(*info))
		return
	}

	//     initialize A to diagonal matrix
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*m); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= minint(*m, *n); i++ {
		a.Set(i-1, i-1, d.Get(i-1))
	}

	//     Quick exit if the user wants a diagonal matrix
	if ((*kl) == 0) && ((*ku) == 0) {
		return
	}

	//     pre- and post-multiply A by random orthogonal matrices
	for i = minint(*m, *n); i >= 1; i-- {
		if i < (*m) {
			//
			//           generate random reflection
			//
			golapack.Dlarnv(func() *int { y := 3; return &y }(), iseed, toPtr((*m)-i+1), work)
			wn = goblas.Dnrm2(toPtr((*m)-i+1), work, toPtr(1))
			wa = math.Copysign(wn, work.Get(0))
			if wn == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				goblas.Dscal(toPtr((*m)-i), toPtrf64(one/wb), work.Off(1), toPtr(1))
				work.Set(0, one)
				tau = wb / wa
			}

			//           multiply A(i:m,i:n) by random reflection from the left
			goblas.Dgemv(mat.Trans, toPtr((*m)-i+1), toPtr((*n)-i+1), &one, a.Off(i-1, i-1), lda, work, toPtr(1), &zero, work.Off((*m)+1-1), toPtr(1))
			goblas.Dger(toPtr((*m)-i+1), toPtr((*n)-i+1), toPtrf64(-tau), work, toPtr(1), work.Off((*m)+1-1), toPtr(1), a.Off(i-1, i-1), lda)
		}
		if i < (*n) {
			//           generate random reflection
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

			//           multiply A(i:m,i:n) by random reflection from the right
			goblas.Dgemv(mat.NoTrans, toPtr((*m)-i+1), toPtr((*n)-i+1), &one, a.Off(i-1, i-1), lda, work, toPtr(1), &zero, work.Off((*n)+1-1), toPtr(1))
			goblas.Dger(toPtr((*m)-i+1), toPtr((*n)-i+1), toPtrf64(-tau), work.Off((*n)+1-1), toPtr(1), work, toPtr(1), a.Off(i-1, i-1), lda)
		}
	}

	//     Reduce number of subdiagonals to KL and number of superdiagonals
	//     to KU
	for i = 1; i <= maxint((*m)-1-(*kl), (*n)-1-(*ku)); i++ {
		if (*kl) <= (*ku) {
			//           annihilate subdiagonal elements first (necessary if KL = 0)
			if i <= minint((*m)-1-(*kl), *n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = goblas.Dnrm2(toPtr((*m)-(*kl)-i+1), a.Vector((*kl)+i-1, i-1), toPtr(1))
				wa = math.Copysign(wn, a.Get((*kl)+i-1, i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get((*kl)+i-1, i-1) + wa
					goblas.Dscal(toPtr((*m)-(*kl)-i), toPtrf64(one/wb), a.Vector((*kl)+i+1-1, i-1), toPtr(1))
					a.Set((*kl)+i-1, i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				goblas.Dgemv(mat.Trans, toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), &one, a.Off((*kl)+i-1, i+1-1), lda, a.Vector((*kl)+i-1, i-1), toPtr(1), &zero, work, toPtr(1))
				goblas.Dger(toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), toPtrf64(-tau), a.Vector((*kl)+i-1, i-1), toPtr(1), work, toPtr(1), a.Off((*kl)+i-1, i+1-1), lda)
				a.Set((*kl)+i-1, i-1, -wa)
			}

			if i <= minint((*n)-1-(*ku), *m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dnrm2(toPtr((*n)-(*ku)-i+1), a.Vector(i-1, (*ku)+i-1), lda)
				wa = math.Copysign(wn, a.Get(i-1, (*ku)+i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, (*ku)+i-1) + wa
					goblas.Dscal(toPtr((*n)-(*ku)-i), toPtrf64(one/wb), a.Vector(i-1, (*ku)+i+1-1), lda)
					a.Set(i-1, (*ku)+i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				goblas.Dgemv(mat.NoTrans, toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), &one, a.Off(i+1-1, (*ku)+i-1), lda, a.Vector(i-1, (*ku)+i-1), lda, &zero, work, toPtr(1))
				goblas.Dger(toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), toPtrf64(-tau), work, toPtr(1), a.Vector(i-1, (*ku)+i-1), lda, a.Off(i+1-1, (*ku)+i-1), lda)
				a.Set(i-1, (*ku)+i-1, -wa)
			}
		} else {
			//           annihilate superdiagonal elements first (necessary if
			//           KU = 0)
			if i <= minint((*n)-1-(*ku), *m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dnrm2(toPtr((*n)-(*ku)-i+1), a.Vector(i-1, (*ku)+i-1), lda)
				wa = math.Copysign(wn, a.Get(i-1, (*ku)+i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, (*ku)+i-1) + wa
					goblas.Dscal(toPtr((*n)-(*ku)-i), toPtrf64(one/wb), a.Vector(i-1, (*ku)+i+1-1), lda)
					a.Set(i-1, (*ku)+i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				goblas.Dgemv(mat.NoTrans, toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), &one, a.Off(i+1-1, (*ku)+i-1), lda, a.Vector(i-1, (*ku)+i-1), lda, &zero, work, toPtr(1))
				goblas.Dger(toPtr((*m)-i), toPtr((*n)-(*ku)-i+1), toPtrf64(-tau), work, toPtr(1), a.Vector(i-1, (*ku)+i-1), lda, a.Off(i+1-1, (*ku)+i-1), lda)
				a.Set(i-1, (*ku)+i-1, -wa)
			}

			if i <= minint((*m)-1-(*kl), *n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = goblas.Dnrm2(toPtr((*m)-(*kl)-i+1), a.Vector((*kl)+i-1, i-1), toPtr(1))
				wa = math.Copysign(wn, a.Get((*kl)+i-1, i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get((*kl)+i-1, i-1) + wa
					goblas.Dscal(toPtr((*m)-(*kl)-i), toPtrf64(one/wb), a.Vector((*kl)+i+1-1, i-1), toPtr(1))
					a.Set((*kl)+i-1, i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				goblas.Dgemv(mat.Trans, toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), &one, a.Off((*kl)+i-1, i+1-1), lda, a.Vector((*kl)+i-1, i-1), toPtr(1), &zero, work, toPtr(1))
				goblas.Dger(toPtr((*m)-(*kl)-i+1), toPtr((*n)-i), toPtrf64(-tau), a.Vector((*kl)+i-1, i-1), toPtr(1), work, toPtr(1), a.Off((*kl)+i-1, i+1-1), lda)
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
