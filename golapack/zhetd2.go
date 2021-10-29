package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetd2 reduces a complex Hermitian matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q**H * A * Q = T.
func Zhetd2(uplo mat.MatUplo, n int, a *mat.CMatrix, d, e *mat.Vector, tau *mat.CVector) (err error) {
	var upper bool
	var alpha, half, one, taui, zero complex128
	var i int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Test the input parameters
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zhetd2", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A
		a.Set(n-1, n-1, a.GetReCmplx(n-1, n-1))
		for i = n - 1; i >= 1; i-- { //
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(1:i-1,i+1)
			alpha = a.Get(i-1, i)
			alpha, taui = Zlarfg(i, alpha, a.CVector(0, i, 1))
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				a.Set(i-1, i, one)

				//              Compute  x := tau * A * v  storing x in TAU(1:i)
				if err = goblas.Zhemv(uplo, i, taui, a, a.CVector(0, i, 1), zero, tau.Off(0, 1)); err != nil {
					panic(err)
				}

				//              Compute  w := x - 1/2 * tau * (x**H * v) * v
				alpha = -half * taui * goblas.Zdotc(i, tau.Off(0, 1), a.CVector(0, i, 1))
				goblas.Zaxpy(i, alpha, a.CVector(0, i, 1), tau.Off(0, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				if err = goblas.Zher2(uplo, i, -one, a.CVector(0, i, 1), tau.Off(0, 1), a); err != nil {
					panic(err)
				}

			} else {
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			}
			a.Set(i-1, i, e.GetCmplx(i-1))
			d.Set(i, a.GetRe(i, i))
			tau.Set(i-1, taui)
		}
		d.Set(0, a.GetRe(0, 0))
	} else {
		//        Reduce the lower triangle of A
		a.Set(0, 0, a.GetReCmplx(0, 0))
		for i = 1; i <= n-1; i++ {
			//           Generate elementary reflector H(i) = I - tau * v * v**H
			//           to annihilate A(i+2:n,i)
			alpha = a.Get(i, i-1)
			alpha, taui = Zlarfg(n-i, alpha, a.CVector(min(i+2, n)-1, i-1, 1))
			e.Set(i-1, real(alpha))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				a.Set(i, i-1, one)

				//              Compute  x := tau * A * v  storing y in TAU(i:n-1)
				if err = goblas.Zhemv(uplo, n-i, taui, a.Off(i, i), a.CVector(i, i-1, 1), zero, tau.Off(i-1, 1)); err != nil {
					panic(err)
				}

				//              Compute  w := x - 1/2 * tau * (x**H * v) * v
				alpha = -half * taui * goblas.Zdotc(n-i, tau.Off(i-1, 1), a.CVector(i, i-1, 1))
				goblas.Zaxpy(n-i, alpha, a.CVector(i, i-1, 1), tau.Off(i-1, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**H - w * v**H
				if err = goblas.Zher2(uplo, n-i, -one, a.CVector(i, i-1, 1), tau.Off(i-1, 1), a.Off(i, i)); err != nil {
					panic(err)
				}

			} else {
				a.Set(i, i, a.GetReCmplx(i, i))
			}
			a.Set(i, i-1, e.GetCmplx(i-1))
			d.Set(i-1, a.GetRe(i-1, i-1))
			tau.Set(i-1, taui)
		}
		d.Set(n-1, a.GetRe(n-1, n-1))
	}

	return
}
