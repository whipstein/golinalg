package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Zlatrd reduces NB rows and columns of a complex Hermitian matrix A to
// Hermitian tridiagonal form by a unitary similarity
// transformation Q**H * A * Q, and returns the matrices V and W which are
// needed to apply the transformation to the unreduced part of A.
//
// If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a
// matrix, of which the upper triangle is supplied;
// if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a
// matrix, of which the lower triangle is supplied.
//
// This is an auxiliary routine called by ZHETRD.
func Zlatrd(uplo mat.MatUplo, n, nb int, a *mat.CMatrix, e *mat.Vector, tau *mat.CVector, w *mat.CMatrix) {
	var alpha, half, one, zero complex128
	var i, iw int
	var err error

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Quick return if possible
	if n <= 0 {
		return
	}

	if uplo == Upper {
		//        Reduce last NB columns of upper triangle
		for i = n; i >= n-nb+1; i-- {
			iw = i - n + nb
			if i < n {
				//              Update A(1:i,i)
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
				Zlacgv(n-i, w.Off(i-1, iw).CVector(), w.Rows)
				if err = a.Off(0, i-1).CVector().Gemv(NoTrans, i, n-i, -one, a.Off(0, i), w.Off(i-1, iw).CVector(), w.Rows, one, 1); err != nil {
					panic(err)
				}
				Zlacgv(n-i, w.Off(i-1, iw).CVector(), w.Rows)
				Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
				if err = a.Off(0, i-1).CVector().Gemv(NoTrans, i, n-i, -one, w.Off(0, iw), a.Off(i-1, i).CVector(), a.Rows, one, 1); err != nil {
					panic(err)
				}
				Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			}
			if i > 1 {
				//              Generate elementary reflector H(i) to annihilate
				//              A(1:i-2,i)
				alpha = a.Get(i-1-1, i-1)
				alpha, *tau.GetPtr(i - 1 - 1) = Zlarfg(i-1, alpha, a.Off(0, i-1).CVector(), 1)
				e.Set(i-1-1, real(alpha))
				a.Set(i-1-1, i-1, one)

				//              Compute W(1:i-1,i)
				if err = w.Off(0, iw-1).CVector().Hemv(Upper, i-1, one, a, a.Off(0, i-1).CVector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if i < n {
					if err = w.Off(i, iw-1).CVector().Gemv(ConjTrans, i-1, n-i, one, w.Off(0, iw), a.Off(0, i-1).CVector(), 1, zero, 1); err != nil {
						panic(err)
					}
					if err = w.Off(0, iw-1).CVector().Gemv(NoTrans, i-1, n-i, -one, a.Off(0, i), w.Off(i, iw-1).CVector(), 1, one, 1); err != nil {
						panic(err)
					}
					if err = w.Off(i, iw-1).CVector().Gemv(ConjTrans, i-1, n-i, one, a.Off(0, i), a.Off(0, i-1).CVector(), 1, zero, 1); err != nil {
						panic(err)
					}
					if err = w.Off(0, iw-1).CVector().Gemv(NoTrans, i-1, n-i, -one, w.Off(0, iw), w.Off(i, iw-1).CVector(), 1, one, 1); err != nil {
						panic(err)
					}
				}
				w.Off(0, iw-1).CVector().Scal(i-1, tau.Get(i-1-1), 1)
				alpha = -half * tau.Get(i-1-1) * a.Off(0, i-1).CVector().Dotc(i-1, w.Off(0, iw-1).CVector(), 1, 1)
				w.Off(0, iw-1).CVector().Axpy(i-1, alpha, a.Off(0, i-1).CVector(), 1, 1)
			}

		}
	} else {
		//        Reduce first NB columns of lower triangle
		for i = 1; i <= nb; i++ {
			//           Update A(i:n,i)
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			Zlacgv(i-1, w.Off(i-1, 0).CVector(), w.Rows)
			if err = a.Off(i-1, i-1).CVector().Gemv(NoTrans, n-i+1, i-1, -one, a.Off(i-1, 0), w.Off(i-1, 0).CVector(), w.Rows, one, 1); err != nil {
				panic(err)
			}
			Zlacgv(i-1, w.Off(i-1, 0).CVector(), w.Rows)
			Zlacgv(i-1, a.Off(i-1, 0).CVector(), a.Rows)
			if err = a.Off(i-1, i-1).CVector().Gemv(NoTrans, n-i+1, i-1, -one, w.Off(i-1, 0), a.Off(i-1, 0).CVector(), a.Rows, one, 1); err != nil {
				panic(err)
			}
			Zlacgv(i-1, a.Off(i-1, 0).CVector(), a.Rows)
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			if i < n {

				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:n,i)
				alpha = a.Get(i, i-1)
				alpha, *tau.GetPtr(i - 1) = Zlarfg(n-i, alpha, a.Off(min(i+2, n)-1, i-1).CVector(), 1)
				e.Set(i-1, real(alpha))
				a.Set(i, i-1, one)

				//              Compute W(i+1:n,i)
				if err = w.Off(i, i-1).CVector().Hemv(Lower, n-i, one, a.Off(i, i), a.Off(i, i-1).CVector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(0, i-1).CVector().Gemv(ConjTrans, n-i, i-1, one, w.Off(i, 0), a.Off(i, i-1).CVector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(i, i-1).CVector().Gemv(NoTrans, n-i, i-1, -one, a.Off(i, 0), w.Off(0, i-1).CVector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = w.Off(0, i-1).CVector().Gemv(ConjTrans, n-i, i-1, one, a.Off(i, 0), a.Off(i, i-1).CVector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(i, i-1).CVector().Gemv(NoTrans, n-i, i-1, -one, w.Off(i, 0), w.Off(0, i-1).CVector(), 1, one, 1); err != nil {
					panic(err)
				}
				w.Off(i, i-1).CVector().Scal(n-i, tau.Get(i-1), 1)
				alpha = -half * tau.Get(i-1) * a.Off(i, i-1).CVector().Dotc(n-i, w.Off(i, i-1).CVector(), 1, 1)
				w.Off(i, i-1).CVector().Axpy(n-i, alpha, a.Off(i, i-1).CVector(), 1, 1)
			}

		}
	}
}
