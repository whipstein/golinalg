package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dlagv2(a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, alphar, alphai, beta *mat.Vector, csl, snl, csr, snr *float64) {
	var anorm, ascale, bnorm, bscale, h1, h2, h3, one, qq, r, rr, safmin, scale1, scale2, t, ulp, wi, wr1, wr2, zero float64

	zero = 0.0
	one = 1.0

	safmin = Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)

	//     Scale A
	anorm = maxf64(math.Abs(a.Get(0, 0))+math.Abs(a.Get(1, 0)), math.Abs(a.Get(0, 1))+math.Abs(a.Get(1, 1)), safmin)
	ascale = one / anorm
	a.Set(0, 0, ascale*a.Get(0, 0))
	a.Set(0, 1, ascale*a.Get(0, 1))
	a.Set(1, 0, ascale*a.Get(1, 0))
	a.Set(1, 1, ascale*a.Get(1, 1))

	//     Scale B
	bnorm = maxf64(math.Abs(b.Get(0, 0)), math.Abs(b.Get(0, 1))+math.Abs(b.Get(1, 1)), safmin)
	bscale = one / bnorm
	b.Set(0, 0, bscale*b.Get(0, 0))
	b.Set(0, 1, bscale*b.Get(0, 1))
	b.Set(1, 1, bscale*b.Get(1, 1))

	//     Check if A can be deflated
	if math.Abs(a.Get(1, 0)) <= ulp {
		(*csl) = one
		(*snl) = zero
		(*csr) = one
		(*snr) = zero
		a.Set(1, 0, zero)
		b.Set(1, 0, zero)
		wi = zero

		//     Check if B is singular
	} else if math.Abs(b.Get(0, 0)) <= ulp {
		Dlartg(a.GetPtr(0, 0), a.GetPtr(1, 0), csl, snl, &r)
		(*csr) = one
		(*snr) = zero
		goblas.Drot(2, a.Vector(0, 0), *lda, a.Vector(1, 0), *lda, *csl, *snl)
		goblas.Drot(2, b.Vector(0, 0), *ldb, b.Vector(1, 0), *ldb, *csl, *snl)
		a.Set(1, 0, zero)
		b.Set(0, 0, zero)
		b.Set(1, 0, zero)
		wi = zero

	} else if math.Abs(b.Get(1, 1)) <= ulp {
		Dlartg(a.GetPtr(1, 1), a.GetPtr(1, 0), csr, snr, &t)
		(*snr) = -(*snr)
		goblas.Drot(2, a.Vector(0, 0), 1, a.Vector(0, 1), 1, *csr, *snr)
		goblas.Drot(2, b.Vector(0, 0), 1, b.Vector(0, 1), 1, *csr, *snr)
		(*csl) = one
		(*snl) = zero
		a.Set(1, 0, zero)
		b.Set(1, 0, zero)
		b.Set(1, 1, zero)
		wi = zero

	} else {
		//        B is nonsingular, first compute the eigenvalues of (A,B)
		Dlag2(a, lda, b, ldb, &safmin, &scale1, &scale2, &wr1, &wr2, &wi)

		if wi == zero {
			//           two real eigenvalues, compute s*A-w*B
			h1 = scale1*a.Get(0, 0) - wr1*b.Get(0, 0)
			h2 = scale1*a.Get(0, 1) - wr1*b.Get(0, 1)
			h3 = scale1*a.Get(1, 1) - wr1*b.Get(1, 1)

			rr = Dlapy2(&h1, &h2)
			qq = Dlapy2(toPtrf64(scale1*a.Get(1, 0)), &h3)

			if rr > qq {
				//              find right rotation matrix to zero 1,1 element of
				//              (sA - wB)
				Dlartg(&h2, &h1, csr, snr, &t)

			} else {
				//              find right rotation matrix to zero 2,1 element of
				//              (sA - wB)
				Dlartg(&h3, toPtrf64(scale1*a.Get(1, 0)), csr, snr, &t)

			}

			(*snr) = -(*snr)
			goblas.Drot(2, a.Vector(0, 0), 1, a.Vector(0, 1), 1, *csr, *snr)
			goblas.Drot(2, b.Vector(0, 0), 1, b.Vector(0, 1), 1, *csr, *snr)

			//           compute inf norms of A and B
			h1 = maxf64(math.Abs(a.Get(0, 0))+math.Abs(a.Get(0, 1)), math.Abs(a.Get(1, 0))+math.Abs(a.Get(1, 1)))
			h2 = maxf64(math.Abs(b.Get(0, 0))+math.Abs(b.Get(0, 1)), math.Abs(b.Get(1, 0))+math.Abs(b.Get(1, 1)))

			if (scale1 * h1) >= math.Abs(wr1)*h2 {
				//              find left rotation matrix Q to zero out B(2,1)
				Dlartg(b.GetPtr(0, 0), b.GetPtr(1, 0), csl, snl, &r)

			} else {
				//              find left rotation matrix Q to zero out A(2,1)
				Dlartg(a.GetPtr(0, 0), a.GetPtr(1, 0), csl, snl, &r)

			}

			goblas.Drot(2, a.Vector(0, 0), *lda, a.Vector(1, 0), *lda, *csl, *snl)
			goblas.Drot(2, b.Vector(0, 0), *ldb, b.Vector(1, 0), *ldb, *csl, *snl)

			a.Set(1, 0, zero)
			b.Set(1, 0, zero)

		} else {
			//           a pair of complex conjugate eigenvalues
			//           first compute the SVD of the matrix B
			Dlasv2(b.GetPtr(0, 0), b.GetPtr(0, 1), b.GetPtr(1, 1), &r, &t, snr, csr, snl, csl)

			//           Form (A,B) := Q(A,B)Z**T where Q is left rotation matrix and
			//           Z is right rotation matrix computed from DLASV2
			goblas.Drot(2, a.Vector(0, 0), *lda, a.Vector(1, 0), *lda, *csl, *snl)
			goblas.Drot(2, b.Vector(0, 0), *ldb, b.Vector(1, 0), *ldb, *csl, *snl)
			goblas.Drot(2, a.Vector(0, 0), 1, a.Vector(0, 1), 1, *csr, *snr)
			goblas.Drot(2, b.Vector(0, 0), 1, b.Vector(0, 1), 1, *csr, *snr)

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
}
