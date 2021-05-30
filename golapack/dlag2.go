package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlag2 computes the eigenvalues of a 2 x 2 generalized eigenvalue
// problem  A - w B, with scaling as necessary to avoid over-/underflow.
//
// The scaling factor "s" results in a modified eigenvalue equation
//
//     s A - w B
//
// where  s  is a non-negative scaling factor chosen so that  w,  w B,
// and  s A  do not overflow and, if possible, do not underflow, either.
func Dlag2(a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, safmin, scale1, scale2, wr1, wr2, wi *float64) {
	var a11, a12, a21, a22, abi22, anorm, as11, as12, as22, ascale, b11, b12, b22, binv11, binv22, bmin, bnorm, bscale, bsize, c1, c2, c3, c4, c5, diff, discr, fuzzy1, half, one, pp, qq, r, rtmax, rtmin, s1, s2, safmax, shift, ss, sum, two, wabs, wbig, wdet, wscale, wsize, wsmall, zero float64

	zero = 0.0
	one = 1.0
	two = 2.0
	half = one / two
	fuzzy1 = one + 1.0e-5

	rtmin = math.Sqrt(*safmin)
	rtmax = one / rtmin
	safmax = one / (*safmin)

	//     Scale A
	anorm = maxf64(math.Abs(a.Get(0, 0))+math.Abs(a.Get(1, 0)), math.Abs(a.Get(0, 1))+math.Abs(a.Get(1, 1)), *safmin)
	ascale = one / anorm
	a11 = ascale * a.Get(0, 0)
	a21 = ascale * a.Get(1, 0)
	a12 = ascale * a.Get(0, 1)
	a22 = ascale * a.Get(1, 1)

	//     Perturb B if necessary to insure non-singularity
	b11 = b.Get(0, 0)
	b12 = b.Get(0, 1)
	b22 = b.Get(1, 1)
	bmin = rtmin * maxf64(math.Abs(b11), math.Abs(b12), math.Abs(b22), rtmin)
	if math.Abs(b11) < bmin {
		b11 = signf64(bmin, b11)
	}
	if math.Abs(b22) < bmin {
		b22 = signf64(bmin, b22)
	}

	//     Scale B
	bnorm = maxf64(math.Abs(b11), math.Abs(b12)+math.Abs(b22), *safmin)
	bsize = maxf64(math.Abs(b11), math.Abs(b22))
	bscale = one / bsize
	b11 = b11 * bscale
	b12 = b12 * bscale
	b22 = b22 * bscale

	//     Compute larger eigenvalue by method described by C. van Loan
	//
	//     ( AS is A shifted by -SHIFT*B )
	binv11 = one / b11
	binv22 = one / b22
	s1 = a11 * binv11
	s2 = a22 * binv22
	if math.Abs(s1) <= math.Abs(s2) {
		as12 = a12 - s1*b12
		as22 = a22 - s1*b22
		ss = a21 * (binv11 * binv22)
		abi22 = as22*binv22 - ss*b12
		pp = half * abi22
		shift = s1
	} else {
		as12 = a12 - s2*b12
		as11 = a11 - s2*b11
		ss = a21 * (binv11 * binv22)
		abi22 = -ss * b12
		pp = half * (as11*binv11 + abi22)
		shift = s2
	}
	qq = ss * as12
	if math.Abs(pp*rtmin) >= one {
		discr = math.Pow(rtmin*pp, 2) + qq*(*safmin)
		r = math.Sqrt(math.Abs(discr)) * rtmax
	} else {
		if math.Pow(pp, 2)+math.Abs(qq) <= (*safmin) {
			discr = math.Pow(rtmax*pp, 2) + qq*safmax
			r = math.Sqrt(math.Abs(discr)) * rtmin
		} else {
			discr = math.Pow(pp, 2) + qq
			r = math.Sqrt(math.Abs(discr))
		}
	}

	//     Note: the test of R in the following IF is to cover the case when
	//           DISCR is small and negative and is flushed to zero during
	//           the calculation of R.  On machines which have a consistent
	//           flush-to-zero threshold and handle numbers above that
	//           threshold correctly, it would not be necessary.
	if discr >= zero || r == zero {
		sum = pp + signf64(r, pp)
		diff = pp - signf64(r, pp)
		wbig = shift + sum

		//        Compute smaller eigenvalue
		wsmall = shift + diff
		if half*math.Abs(wbig) > maxf64(math.Abs(wsmall), *safmin) {
			wdet = (a11*a22 - a12*a21) * (binv11 * binv22)
			wsmall = wdet / wbig
		}

		//        Choose (real) eigenvalue closest to 2,2 element of A*B**(-1)
		//        for WR1.
		if pp > abi22 {
			(*wr1) = minf64(wbig, wsmall)
			(*wr2) = maxf64(wbig, wsmall)
		} else {
			(*wr1) = maxf64(wbig, wsmall)
			(*wr2) = minf64(wbig, wsmall)
		}
		(*wi) = zero
	} else {
		//        Complex eigenvalues
		(*wr1) = shift + pp
		(*wr2) = (*wr1)
		(*wi) = r
	}

	//     Further scaling to avoid underflow and overflow in computing
	//     SCALE1 and overflow in computing w*B.
	//
	//     This scale factor (WSCALE) is bounded from above using C1 and C2,
	//     and from below using C3 and C4.
	//        C1 implements the condition  s A  must never overflow.
	//        C2 implements the condition  w B  must never overflow.
	//        C3, with C2,
	//           implement the condition that s A - w B must never overflow.
	//        C4 implements the condition  s    should not underflow.
	//        C5 implements the condition  maxf64(s,|w|) should be at least 2.
	c1 = bsize * ((*safmin) * maxf64(one, ascale))
	c2 = (*safmin) * maxf64(one, bnorm)
	c3 = bsize * (*safmin)
	if ascale <= one && bsize <= one {
		c4 = minf64(one, (ascale/(*safmin))*bsize)
	} else {
		c4 = one
	}
	if ascale <= one || bsize <= one {
		c5 = minf64(one, ascale*bsize)
	} else {
		c5 = one
	}

	//     Scale first eigenvalue
	wabs = math.Abs(*wr1) + math.Abs(*wi)
	wsize = maxf64(*safmin, c1, fuzzy1*(wabs*c2+c3), minf64(c4, half*maxf64(wabs, c5)))
	if wsize != one {
		wscale = one / wsize
		if wsize > one {
			(*scale1) = (maxf64(ascale, bsize) * wscale) * minf64(ascale, bsize)
		} else {
			(*scale1) = (minf64(ascale, bsize) * wscale) * maxf64(ascale, bsize)
		}
		(*wr1) = (*wr1) * wscale
		if (*wi) != zero {
			(*wi) = (*wi) * wscale
			(*wr2) = (*wr1)
			(*scale2) = (*scale1)
		}
	} else {
		(*scale1) = ascale * bsize
		(*scale2) = (*scale1)
	}

	//     Scale second eigenvalue (if real)
	if (*wi) == zero {
		wsize = maxf64(*safmin, c1, fuzzy1*(math.Abs(*wr2)*c2+c3), minf64(c4, half*maxf64(math.Abs(*wr2), c5)))
		if wsize != one {
			wscale = one / wsize
			if wsize > one {
				(*scale2) = (maxf64(ascale, bsize) * wscale) * minf64(ascale, bsize)
			} else {
				(*scale2) = (minf64(ascale, bsize) * wscale) * maxf64(ascale, bsize)
			}
			(*wr2) = (*wr2) * wscale
		} else {
			(*scale2) = ascale * bsize
		}
	}
}
