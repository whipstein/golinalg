package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhptrd reduces a complex Hermitian matrix A stored in packed form to
// real symmetric tridiagonal form T by a unitary similarity
// transformation: Q**H * A * Q = T.
func Zhptrd(uplo mat.MatUplo, n int, ap *mat.CVector, d, e *mat.Vector, tau *mat.CVector) (err error) {
	var upper bool
	var alpha, half, one, taui, zero complex128
	var i, i1, i1i1, ii int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Test the input parameters
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Zhptrd", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        I1 is the index in AP of A(1,I+1).
		i1 = n*(n-1)/2 + 1
		ap.Set(i1+n-1-1, ap.GetReCmplx(i1+n-1-1))
		for i = n - 1; i >= 1; i-- { //
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(1:i-1,i+1)
			alpha = ap.Get(i1 + i - 1 - 1)
			alpha, taui = Zlarfg(i, alpha, ap.Off(i1-1, 1))
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				ap.Set(i1+i-1-1, one)

				//              Compute  y := tau * A * v  storing y in TAU(1:i)
				if err = goblas.Zhpmv(uplo, i, taui, ap, ap.Off(i1-1, 1), zero, tau.Off(0, 1)); err != nil {
					panic(err)
				}

				//              Compute  w := y - 1/2 * tau * (y**H *v) * v
				alpha = -half * taui * goblas.Zdotc(i, tau.Off(0, 1), ap.Off(i1-1, 1))
				goblas.Zaxpy(i, alpha, ap.Off(i1-1, 1), tau.Off(0, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				if err = goblas.Zhpr2(uplo, i, -one, ap.Off(i1-1, 1), tau.Off(0, 1), ap); err != nil {
					panic(err)
				}

			}
			ap.Set(i1+i-1-1, e.GetCmplx(i-1))
			d.Set(i, ap.GetRe(i1+i-1))
			tau.Set(i-1, taui)
			i1 = i1 - i
		}
		d.Set(0, ap.GetRe(0))
	} else {
		//        Reduce the lower triangle of A. II is the index in AP of
		//        A(i,i) and I1I1 is the index of A(i+1,i+1).
		ii = 1
		ap.Set(0, ap.GetReCmplx(0))
		for i = 1; i <= n-1; i++ {
			i1i1 = ii + n - i + 1

			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(i+2:n,i)
			alpha = ap.Get(ii + 1 - 1)
			alpha, taui = Zlarfg(n-i, alpha, ap.Off(ii+2-1, 1))
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				ap.Set(ii, one)

				//              Compute  y := tau * A * v  storing y in TAU(i:n-1)
				if err = goblas.Zhpmv(uplo, n-i, taui, ap.Off(i1i1-1), ap.Off(ii, 1), zero, tau.Off(i-1, 1)); err != nil {
					panic(err)
				}

				//              Compute  w := y - 1/2 * tau * (y**H *v) * v
				alpha = -half * taui * goblas.Zdotc(n-i, tau.Off(i-1, 1), ap.Off(ii, 1))
				goblas.Zaxpy(n-i, alpha, ap.Off(ii, 1), tau.Off(i-1, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				if err = goblas.Zhpr2(uplo, n-i, -one, ap.Off(ii, 1), tau.Off(i-1, 1), ap.Off(i1i1-1)); err != nil {
					panic(err)
				}

			}
			ap.Set(ii, e.GetCmplx(i-1))
			d.Set(i-1, ap.GetRe(ii-1))
			tau.Set(i-1, taui)
			ii = i1i1
		}
		d.Set(n-1, ap.GetRe(ii-1))
	}

	return
}
