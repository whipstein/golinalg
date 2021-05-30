package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhptrd reduces a complex Hermitian matrix A stored in packed form to
// real symmetric tridiagonal form T by a unitary similarity
// transformation: Q**H * A * Q = T.
func Zhptrd(uplo byte, n *int, ap *mat.CVector, d, e *mat.Vector, tau *mat.CVector, info *int) {
	var upper bool
	var alpha, half, one, taui, zero complex128
	var i, i1, i1i1, ii int

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
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPTRD"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        I1 is the index in AP of A(1,I+1).
		i1 = (*n)*((*n)-1)/2 + 1
		ap.Set(i1+(*n)-1-1, ap.GetReCmplx(i1+(*n)-1-1))
		for i = (*n) - 1; i >= 1; i-- { //
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(1:i-1,i+1)
			alpha = ap.Get(i1 + i - 1 - 1)
			Zlarfg(&i, &alpha, ap.Off(i1-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				ap.Set(i1+i-1-1, one)

				//              Compute  y := tau * A * v  storing y in TAU(1:i)
				goblas.Zhpmv(mat.UploByte(uplo), &i, &taui, ap, ap.Off(i1-1), func() *int { y := 1; return &y }(), &zero, tau, func() *int { y := 1; return &y }())

				//              Compute  w := y - 1/2 * tau * (y**H *v) * v
				alpha = -half * taui * goblas.Zdotc(&i, tau, func() *int { y := 1; return &y }(), ap.Off(i1-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(&i, &alpha, ap.Off(i1-1), func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }())

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				goblas.Zhpr2(mat.UploByte(uplo), &i, toPtrc128(-one), ap.Off(i1-1), func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), ap)

			}
			ap.Set(i1+i-1-1, e.GetCmplx(i-1))
			d.Set(i+1-1, ap.GetRe(i1+i-1))
			tau.Set(i-1, taui)
			i1 = i1 - i
		}
		d.Set(0, ap.GetRe(0))
	} else {
		//        Reduce the lower triangle of A. II is the index in AP of
		//        A(i,i) and I1I1 is the index of A(i+1,i+1).
		ii = 1
		ap.Set(0, ap.GetReCmplx(0))
		for i = 1; i <= (*n)-1; i++ {
			i1i1 = ii + (*n) - i + 1

			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(i+2:n,i)
			alpha = ap.Get(ii + 1 - 1)
			Zlarfg(toPtr((*n)-i), &alpha, ap.Off(ii+2-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				ap.Set(ii+1-1, one)

				//              Compute  y := tau * A * v  storing y in TAU(i:n-1)
				goblas.Zhpmv(mat.UploByte(uplo), toPtr((*n)-i), &taui, ap.Off(i1i1-1), ap.Off(ii+1-1), func() *int { y := 1; return &y }(), &zero, tau.Off(i-1), func() *int { y := 1; return &y }())

				//              Compute  w := y - 1/2 * tau * (y**H *v) * v
				alpha = -half * taui * goblas.Zdotc(toPtr((*n)-i), tau.Off(i-1), func() *int { y := 1; return &y }(), ap.Off(ii+1-1), func() *int { y := 1; return &y }())
				goblas.Zaxpy(toPtr((*n)-i), &alpha, ap.Off(ii+1-1), func() *int { y := 1; return &y }(), tau.Off(i-1), func() *int { y := 1; return &y }())

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				goblas.Zhpr2(mat.UploByte(uplo), toPtr((*n)-i), toPtrc128(-one), ap.Off(ii+1-1), func() *int { y := 1; return &y }(), tau.Off(i-1), func() *int { y := 1; return &y }(), ap.Off(i1i1-1))

			}
			ap.Set(ii+1-1, e.GetCmplx(i-1))
			d.Set(i-1, ap.GetRe(ii-1))
			tau.Set(i-1, taui)
			ii = i1i1
		}
		d.Set((*n)-1, ap.GetRe(ii-1))
	}
}
