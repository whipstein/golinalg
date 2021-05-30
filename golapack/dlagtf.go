package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlagtf factorizes the matrix (T - lambda*I), where T is an n by n
// tridiagonal matrix and lambda is a scalar, as
//
//    T - lambda*I = PLU,
//
// where P is a permutation matrix, L is a unit lower tridiagonal matrix
// with at most one non-zero sub-diagonal elements per column and U is
// an upper triangular matrix with at most two non-zero super-diagonal
// elements per column.
//
// The factorization is obtained by Gaussian elimination with partial
// pivoting and implicit row scaling.
//
// The parameter LAMBDA is included in the routine so that DLAGTF may
// be used, in conjunction with DLAGTS, to obtain eigenvectors of T by
// inverse iteration.
func Dlagtf(n *int, a *mat.Vector, lambda *float64, b, c *mat.Vector, tol *float64, d *mat.Vector, in *[]int, info *int) {
	var eps, mult, piv1, piv2, scale1, scale2, temp, tl, zero float64
	var k int

	zero = 0.0

	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("DLAGTF"), -(*info))
		return
	}

	if (*n) == 0 {
		return
	}

	a.Set(0, a.Get(0)-(*lambda))
	(*in)[(*n)-1] = 0
	if (*n) == 1 {
		if a.Get(0) == zero {
			(*in)[0] = 1
		}
		return
	}

	eps = Dlamch(Epsilon)

	tl = maxf64(*tol, eps)
	scale1 = math.Abs(a.Get(0)) + math.Abs(b.Get(0))
	for k = 1; k <= (*n)-1; k++ {
		a.Set(k+1-1, a.Get(k+1-1)-(*lambda))
		scale2 = math.Abs(c.Get(k-1)) + math.Abs(a.Get(k+1-1))
		if k < ((*n) - 1) {
			scale2 = scale2 + math.Abs(b.Get(k+1-1))
		}
		if a.Get(k-1) == zero {
			piv1 = zero
		} else {
			piv1 = math.Abs(a.Get(k-1)) / scale1
		}
		if c.Get(k-1) == zero {
			(*in)[k-1] = 0
			piv2 = zero
			scale1 = scale2
			if k < ((*n) - 1) {
				d.Set(k-1, zero)
			}
		} else {
			piv2 = math.Abs(c.Get(k-1)) / scale2
			if piv2 <= piv1 {
				(*in)[k-1] = 0
				scale1 = scale2
				c.Set(k-1, c.Get(k-1)/a.Get(k-1))
				a.Set(k+1-1, a.Get(k+1-1)-c.Get(k-1)*b.Get(k-1))
				if k < ((*n) - 1) {
					d.Set(k-1, zero)
				}
			} else {
				(*in)[k-1] = 1
				mult = a.Get(k-1) / c.Get(k-1)
				a.Set(k-1, c.Get(k-1))
				temp = a.Get(k + 1 - 1)
				a.Set(k+1-1, b.Get(k-1)-mult*temp)
				if k < ((*n) - 1) {
					d.Set(k-1, b.Get(k+1-1))
					b.Set(k+1-1, -mult*d.Get(k-1))
				}
				b.Set(k-1, temp)
				c.Set(k-1, mult)
			}
		}
		if (maxf64(piv1, piv2) <= tl) && ((*in)[(*n)-1] == 0) {
			(*in)[(*n)-1] = k
		}
	}
	if (math.Abs(a.Get((*n)-1)) <= scale1*tl) && ((*in)[(*n)-1] == 0) {
		(*in)[(*n)-1] = (*n)
	}
}
