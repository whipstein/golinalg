package golapack

import (
	"github.com/whipstein/golinalg/goblas"
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
				Zlacgv(n-i, w.CVector(i-1, iw))
				if err = goblas.Zgemv(NoTrans, i, n-i, -one, a.Off(0, i), w.CVector(i-1, iw), one, a.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				Zlacgv(n-i, w.CVector(i-1, iw))
				Zlacgv(n-i, a.CVector(i-1, i))
				if err = goblas.Zgemv(NoTrans, i, n-i, -one, w.Off(0, iw), a.CVector(i-1, i), one, a.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				Zlacgv(n-i, a.CVector(i-1, i))
				a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			}
			if i > 1 {
				//              Generate elementary reflector H(i) to annihilate
				//              A(1:i-2,i)
				alpha = a.Get(i-1-1, i-1)
				alpha, *tau.GetPtr(i - 1 - 1) = Zlarfg(i-1, alpha, a.CVector(0, i-1, 1))
				e.Set(i-1-1, real(alpha))
				a.Set(i-1-1, i-1, one)

				//              Compute W(1:i-1,i)
				if err = goblas.Zhemv(Upper, i-1, one, a, a.CVector(0, i-1, 1), zero, w.CVector(0, iw-1, 1)); err != nil {
					panic(err)
				}
				if i < n {
					if err = goblas.Zgemv(ConjTrans, i-1, n-i, one, w.Off(0, iw), a.CVector(0, i-1, 1), zero, w.CVector(i, iw-1, 1)); err != nil {
						panic(err)
					}
					if err = goblas.Zgemv(NoTrans, i-1, n-i, -one, a.Off(0, i), w.CVector(i, iw-1, 1), one, w.CVector(0, iw-1, 1)); err != nil {
						panic(err)
					}
					if err = goblas.Zgemv(ConjTrans, i-1, n-i, one, a.Off(0, i), a.CVector(0, i-1, 1), zero, w.CVector(i, iw-1, 1)); err != nil {
						panic(err)
					}
					if err = goblas.Zgemv(NoTrans, i-1, n-i, -one, w.Off(0, iw), w.CVector(i, iw-1, 1), one, w.CVector(0, iw-1, 1)); err != nil {
						panic(err)
					}
				}
				goblas.Zscal(i-1, tau.Get(i-1-1), w.CVector(0, iw-1, 1))
				alpha = -half * tau.Get(i-1-1) * goblas.Zdotc(i-1, w.CVector(0, iw-1, 1), a.CVector(0, i-1, 1))
				goblas.Zaxpy(i-1, alpha, a.CVector(0, i-1, 1), w.CVector(0, iw-1, 1))
			}

		}
	} else {
		//        Reduce first NB columns of lower triangle
		for i = 1; i <= nb; i++ {
			//           Update A(i:n,i)
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			Zlacgv(i-1, w.CVector(i-1, 0))
			if err = goblas.Zgemv(NoTrans, n-i+1, i-1, -one, a.Off(i-1, 0), w.CVector(i-1, 0), one, a.CVector(i-1, i-1, 1)); err != nil {
				panic(err)
			}
			Zlacgv(i-1, w.CVector(i-1, 0))
			Zlacgv(i-1, a.CVector(i-1, 0))
			if err = goblas.Zgemv(NoTrans, n-i+1, i-1, -one, w.Off(i-1, 0), a.CVector(i-1, 0), one, a.CVector(i-1, i-1, 1)); err != nil {
				panic(err)
			}
			Zlacgv(i-1, a.CVector(i-1, 0))
			a.Set(i-1, i-1, a.GetReCmplx(i-1, i-1))
			if i < n {

				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:n,i)
				alpha = a.Get(i, i-1)
				alpha, *tau.GetPtr(i - 1) = Zlarfg(n-i, alpha, a.CVector(min(i+2, n)-1, i-1, 1))
				e.Set(i-1, real(alpha))
				a.Set(i, i-1, one)

				//              Compute W(i+1:n,i)
				if err = goblas.Zhemv(Lower, n-i, one, a.Off(i, i), a.CVector(i, i-1, 1), zero, w.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, n-i, i-1, one, w.Off(i, 0), a.CVector(i, i-1, 1), zero, w.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, n-i, i-1, -one, a.Off(i, 0), w.CVector(0, i-1, 1), one, w.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, n-i, i-1, one, a.Off(i, 0), a.CVector(i, i-1, 1), zero, w.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, n-i, i-1, -one, w.Off(i, 0), w.CVector(0, i-1, 1), one, w.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				goblas.Zscal(n-i, tau.Get(i-1), w.CVector(i, i-1, 1))
				alpha = -half * tau.Get(i-1) * goblas.Zdotc(n-i, w.CVector(i, i-1, 1), a.CVector(i, i-1, 1))
				goblas.Zaxpy(n-i, alpha, a.CVector(i, i-1, 1), w.CVector(i, i-1, 1))
			}

		}
	}
}
