package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbequ computes row and column scalings intended to equilibrate an
// M-by-N band matrix A and reduce its condition number.  R returns the
// row scale factors and C the column scale factors, chosen to try to
// make the largest element in each row and column of the matrix B with
// elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.
//
// R(i) and C(j) are restricted to be between SMLNUM = smallest safe
// number and BIGNUM = largest safe number.  Use of these scaling
// factors is not guaranteed to reduce the condition number of A but
// works well in practice.
func Zgbequ(m, n, kl, ku *int, ab *mat.CMatrix, ldab *int, r, c *mat.Vector, rowcnd, colcnd, amax *float64, info *int) {
	var bignum, one, rcmax, rcmin, smlnum, zero float64
	var i, j, kd int

	one = 1.0
	zero = 0.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kl) < 0 {
		(*info) = -3
	} else if (*ku) < 0 {
		(*info) = -4
	} else if (*ldab) < (*kl)+(*ku)+1 {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBEQU"), -(*info))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		(*rowcnd) = one
		(*colcnd) = one
		(*amax) = zero
		return
	}

	//     Get machine constants.
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum

	//     Compute row scale factors.
	for i = 1; i <= (*m); i++ {
		r.Set(i-1, zero)
	}

	//     Find the maximum element in each row.
	kd = (*ku) + 1
	for j = 1; j <= (*n); j++ {
		for i = maxint(j-(*ku), 1); i <= minint(j+(*kl), *m); i++ {
			r.Set(i-1, maxf64(r.Get(i-1), Cabs1(ab.Get(kd+i-j-1, j-1))))
		}
	}

	//     Find the maximum and minimum scale factors.
	rcmin = bignum
	rcmax = zero
	for i = 1; i <= (*m); i++ {
		rcmax = maxf64(rcmax, r.Get(i-1))
		rcmin = minf64(rcmin, r.Get(i-1))
	}
	(*amax) = rcmax

	if rcmin == zero {
		//        Find the first zero scale factor and return an error code.
		for i = 1; i <= (*m); i++ {
			if r.Get(i-1) == zero {
				(*info) = i
				return
			}
		}
	} else {
		//        Invert the scale factors.
		for i = 1; i <= (*m); i++ {
			r.Set(i-1, one/minf64(maxf64(r.Get(i-1), smlnum), bignum))
		}

		//        Compute ROWCND = min(R(I)) / max(R(I))
		(*rowcnd) = maxf64(rcmin, smlnum) / minf64(rcmax, bignum)
	}

	//     Compute column scale factors
	for j = 1; j <= (*n); j++ {
		c.Set(j-1, zero)
	}

	//     Find the maximum element in each column,
	//     assuming the row scaling computed above.
	kd = (*ku) + 1
	for j = 1; j <= (*n); j++ {
		for i = maxint(j-(*ku), 1); i <= minint(j+(*kl), *m); i++ {
			c.Set(j-1, maxf64(c.Get(j-1), Cabs1(ab.Get(kd+i-j-1, j-1))*r.Get(i-1)))
		}
	}

	//     Find the maximum and minimum scale factors.
	rcmin = bignum
	rcmax = zero
	for j = 1; j <= (*n); j++ {
		rcmin = minf64(rcmin, c.Get(j-1))
		rcmax = maxf64(rcmax, c.Get(j-1))
	}

	if rcmin == zero {
		//        Find the first zero scale factor and return an error code.
		for j = 1; j <= (*n); j++ {
			if c.Get(j-1) == zero {
				(*info) = (*m) + j
				return
			}
		}
	} else {
		//        Invert the scale factors.
		for j = 1; j <= (*n); j++ {
			c.Set(j-1, one/minf64(maxf64(c.Get(j-1), smlnum), bignum))
		}

		//        Compute COLCND = min(C(J)) / max(C(J))
		(*colcnd) = maxf64(rcmin, smlnum) / minf64(rcmax, bignum)
	}
}
