package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlagv2 computes the Generalized Schur factorization of a real 2-by-2
// matrix pencil (A,B) where B is upper triangular. This routine
// computes orthogonal (rotation) matrices given by CSL, SNL and CSR,
// SNR such that
//
// 1) if the pencil (A,B) has two real eigenvalues (include 0/0 or 1/0
//    types), then
//
//    [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ]
//    [  0  a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ]
//
//    [ b11 b12 ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ]
//    [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ],
//
// 2) if the pencil (A,B) has a pair of complex conjugate eigenvalues,
//    then
//
//    [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ]
//    [ a21 a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ]
//
//    [ b11  0  ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ]
//    [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ]
//
//    where b11 >= b22 > 0.
func Dlagv2(a, b *mat.Matrix, alphar, alphai, beta *mat.Vector) (csl, snl, csr, snr float64) {
	var anorm, ascale, bnorm, bscale, h1, h2, h3, one, qq, rr, safmin, scale1, ulp, wi, wr1, zero float64

	zero = 0.0
	one = 1.0

	safmin = Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)

	//     Scale A
	anorm = math.Max(math.Abs(a.Get(0, 0))+math.Abs(a.Get(1, 0)), math.Max(math.Abs(a.Get(0, 1))+math.Abs(a.Get(1, 1)), safmin))
	ascale = one / anorm
	a.Set(0, 0, ascale*a.Get(0, 0))
	a.Set(0, 1, ascale*a.Get(0, 1))
	a.Set(1, 0, ascale*a.Get(1, 0))
	a.Set(1, 1, ascale*a.Get(1, 1))

	//     Scale B
	bnorm = math.Max(math.Abs(b.Get(0, 0)), math.Max(math.Abs(b.Get(0, 1))+math.Abs(b.Get(1, 1)), safmin))
	bscale = one / bnorm
	b.Set(0, 0, bscale*b.Get(0, 0))
	b.Set(0, 1, bscale*b.Get(0, 1))
	b.Set(1, 1, bscale*b.Get(1, 1))

	//     Check if A can be deflated
	if math.Abs(a.Get(1, 0)) <= ulp {
		csl = one
		snl = zero
		csr = one
		snr = zero
		a.Set(1, 0, zero)
		b.Set(1, 0, zero)
		wi = zero

		//     Check if B is singular
	} else if math.Abs(b.Get(0, 0)) <= ulp {
		csl, snl, _ = Dlartg(a.Get(0, 0), a.Get(1, 0))
		csr = one
		snr = zero
		a.Off(1, 0).Vector().Rot(2, a.Off(0, 0).Vector(), a.Rows, a.Rows, csl, snl)
		b.Off(1, 0).Vector().Rot(2, b.Off(0, 0).Vector(), b.Rows, b.Rows, csl, snl)
		a.Set(1, 0, zero)
		b.Set(0, 0, zero)
		b.Set(1, 0, zero)
		wi = zero

	} else if math.Abs(b.Get(1, 1)) <= ulp {
		csr, snr, _ = Dlartg(a.Get(1, 1), a.Get(1, 0))
		snr = -snr
		a.Off(0, 1).Vector().Rot(2, a.Off(0, 0).Vector(), 1, 1, csr, snr)
		b.Off(0, 1).Vector().Rot(2, b.Off(0, 0).Vector(), 1, 1, csr, snr)
		csl = one
		snl = zero
		a.Set(1, 0, zero)
		b.Set(1, 0, zero)
		b.Set(1, 1, zero)
		wi = zero

	} else {
		//        B is nonsingular, first compute the eigenvalues of (A,B)
		scale1, _, wr1, _, wi = Dlag2(a, b, safmin)

		if wi == zero {
			//           two real eigenvalues, compute s*A-w*B
			h1 = scale1*a.Get(0, 0) - wr1*b.Get(0, 0)
			h2 = scale1*a.Get(0, 1) - wr1*b.Get(0, 1)
			h3 = scale1*a.Get(1, 1) - wr1*b.Get(1, 1)

			rr = Dlapy2(h1, h2)
			qq = Dlapy2(scale1*a.Get(1, 0), h3)

			if rr > qq {
				//              find right rotation matrix to zero 1,1 element of
				//              (sA - wB)
				csr, snr, _ = Dlartg(h2, h1)

			} else {
				//              find right rotation matrix to zero 2,1 element of
				//              (sA - wB)
				csr, snr, _ = Dlartg(h3, scale1*a.Get(1, 0))

			}

			snr = -snr
			a.Off(0, 1).Vector().Rot(2, a.Off(0, 0).Vector(), 1, 1, csr, snr)
			b.Off(0, 1).Vector().Rot(2, b.Off(0, 0).Vector(), 1, 1, csr, snr)

			//           compute inf norms of A and B
			h1 = math.Max(math.Abs(a.Get(0, 0))+math.Abs(a.Get(0, 1)), math.Abs(a.Get(1, 0))+math.Abs(a.Get(1, 1)))
			h2 = math.Max(math.Abs(b.Get(0, 0))+math.Abs(b.Get(0, 1)), math.Abs(b.Get(1, 0))+math.Abs(b.Get(1, 1)))

			if (scale1 * h1) >= math.Abs(wr1)*h2 {
				//              find left rotation matrix Q to zero out B(2,1)
				csl, snl, _ = Dlartg(b.Get(0, 0), b.Get(1, 0))

			} else {
				//              find left rotation matrix Q to zero out A(2,1)
				csl, snl, _ = Dlartg(a.Get(0, 0), a.Get(1, 0))

			}

			a.Off(1, 0).Vector().Rot(2, a.Off(0, 0).Vector(), a.Rows, a.Rows, csl, snl)
			b.Off(1, 0).Vector().Rot(2, b.Off(0, 0).Vector(), b.Rows, b.Rows, csl, snl)

			a.Set(1, 0, zero)
			b.Set(1, 0, zero)

		} else {
			//           a pair of complex conjugate eigenvalues
			//           first compute the SVD of the matrix B
			_, _, snr, csr, snl, csl = Dlasv2(b.Get(0, 0), b.Get(0, 1), b.Get(1, 1))

			//           Form (A,B) := Q(A,B)Z**T where Q is left rotation matrix and
			//           Z is right rotation matrix computed from DLASV2
			a.Off(1, 0).Vector().Rot(2, a.Off(0, 0).Vector(), a.Rows, a.Rows, csl, snl)
			b.Off(1, 0).Vector().Rot(2, b.Off(0, 0).Vector(), b.Rows, b.Rows, csl, snl)
			a.Off(0, 1).Vector().Rot(2, a.Off(0, 0).Vector(), 1, 1, csr, snr)
			b.Off(0, 1).Vector().Rot(2, b.Off(0, 0).Vector(), 1, 1, csr, snr)

			b.Set(1, 0, zero)
			b.Set(0, 1, zero)

		}

	}

	//     Unscaling
	a.Set(0, 0, anorm*a.Get(0, 0))
	a.Set(1, 0, anorm*a.Get(1, 0))
	a.Set(0, 1, anorm*a.Get(0, 1))
	a.Set(1, 1, anorm*a.Get(1, 1))
	b.Set(0, 0, bnorm*b.Get(0, 0))
	b.Set(1, 0, bnorm*b.Get(1, 0))
	b.Set(0, 1, bnorm*b.Get(0, 1))
	b.Set(1, 1, bnorm*b.Get(1, 1))

	if wi == zero {
		alphar.Set(0, a.Get(0, 0))
		alphar.Set(1, a.Get(1, 1))
		alphai.Set(0, zero)
		alphai.Set(1, zero)
		beta.Set(0, b.Get(0, 0))
		beta.Set(1, b.Get(1, 1))
	} else {
		alphar.Set(0, anorm*wr1/scale1/bnorm)
		alphai.Set(0, anorm*wi/scale1/bnorm)
		alphar.Set(1, alphar.Get(0))
		alphai.Set(1, -alphai.Get(0))
		beta.Set(0, one)
		beta.Set(1, one)
	}

	return
}
