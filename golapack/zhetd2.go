package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhetd2 reduces a complex Hermitian matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q**H * A * Q = T.
func Zhetd2(uplo byte, n *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, tau *mat.CVector, info *int) {
	var upper bool
	var alpha, half, one, taui, zero complex128
	var i int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETD2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A
		a.Set((*n)-1, (*n)-1, a.GetReCmplx((*n)-1, (*n)-1))
		for i = (*n) - 1; i >= 1; i-- { //
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(1:i-1,i+1)
			alpha = a.Get(i-1, i+1-1)
			Zlarfg(&i, &alpha, a.CVector(0, i+1-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				a.Set(i-1, i+1-1, one)

				//              Compute  x := tau * A * v  storing x in TAU(1:i)
				goblas.Zhemv(mat.UploByte(uplo), &i, &taui, a, lda, a.CVector(0, i+1-1), func() *int { y := 1; return &y }(), &zero, tau, func() *int { y := 1; return &y }())

				//              Compute  w := x - 1/2 * tau * (x**H * v) * v
				alpha = -half * taui * goblas.Zdotc(&i, tau, func() *int { y := 1; return &y }(), a.CVector(0, i+1-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(&i, &alpha, a.CVector(0, i+1-1), func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }())

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				goblas.Zher2(mat.UploByte(uplo), &i, toPtrc128(-one), a.CVector(0, i+1-1), func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), a, lda)

			} else {
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			}
			a.Set(i-1, i+1-1, e.GetCmplx(i-1))
			d.Set(i+1-1, a.GetRe(i+1-1, i+1-1))
			tau.Set(i-1, taui)
		}
		d.Set(0, a.GetRe(0, 0))
	} else {
		//        Reduce the lower triangle of A
		a.Set(0, 0, a.GetReCmplx(0, 0))
		for i = 1; i <= (*n)-1; i++ {
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(i+2:n,i)
			alpha = a.Get(i+1-1, i-1)
			Zlarfg(toPtr((*n)-i), &alpha, a.CVector(minint(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				a.Set(i+1-1, i-1, one)

				//              Compute  x := tau * A * v  storing y in TAU(i:n-1)
				goblas.Zhemv(mat.UploByte(uplo), toPtr((*n)-i), &taui, a.Off(i+1-1, i+1-1), lda, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &zero, tau.Off(i-1), func() *int { y := 1; return &y }())

				//              Compute  w := x - 1/2 * tau * (x**H * v) * v
				alpha = -half * taui * goblas.Zdotc(toPtr((*n)-i), tau.Off(i-1), func() *int { y := 1; return &y }(), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(toPtr((*n)-i), &alpha, a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), tau.Off(i-1), func() *int { y := 1; return &y }())

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				goblas.Zher2(mat.UploByte(uplo), toPtr((*n)-i), toPtrc128(-one), a.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), tau.Off(i-1), func() *int { y := 1; return &y }(), a.Off(i+1-1, i+1-1), lda)

			} else {
				a.Set(i+1-1, i+1-1, a.GetReCmplx(i+1-1, i+1-1))
			}
			a.Set(i+1-1, i-1, e.GetCmplx(i-1))
			d.Set(i-1, a.GetRe(i-1, i-1))
			tau.Set(i-1, taui)
		}
		d.Set((*n)-1, a.GetRe((*n)-1, (*n)-1))
	}
}
