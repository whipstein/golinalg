package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dlatrd reduces NB rows and columns of a real symmetric matrix A to
// symmetric tridiagonal form by an orthogonal similarity
// transformation Q**T * A * Q, and returns the matrices V and W which are
// needed to apply the transformation to the unreduced part of A.
//
// If UPLO = 'U', DLATRD reduces the last NB rows and columns of a
// matrix, of which the upper triangle is supplied;
// if UPLO = 'L', DLATRD reduces the first NB rows and columns of a
// matrix, of which the lower triangle is supplied.
//
// This is an auxiliary routine called by DSYTRD.
func Dlatrd(uplo mat.MatUplo, n, nb int, a *mat.Matrix, e, tau *mat.Vector, w *mat.Matrix) {
	var alpha, half, one, zero float64
	var i, iw int
	var err error

	zero = 0.0
	one = 1.0
	half = 0.5

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
				if err = a.Off(0, i-1).Vector().Gemv(NoTrans, i, n-i, -one, a.Off(0, i), w.Off(i-1, iw).Vector(), w.Rows, one, 1); err != nil {
					panic(err)
				}
				if err = a.Off(0, i-1).Vector().Gemv(NoTrans, i, n-i, -one, w.Off(0, iw), a.Off(i-1, i).Vector(), a.Rows, one, 1); err != nil {
					panic(err)
				}
			}
			if i > 1 {
				//              Generate elementary reflector H(i) to annihilate
				//              A(1:i-2,i)
				*a.GetPtr(i-1-1, i-1), *tau.GetPtr(i - 1 - 1) = Dlarfg(i-1, a.Get(i-1-1, i-1), a.Off(0, i-1).Vector(), 1)
				e.Set(i-1-1, a.Get(i-1-1, i-1))
				a.Set(i-1-1, i-1, one)

				//              Compute W(1:i-1,i)
				if err = w.Off(0, iw-1).Vector().Symv(Upper, i-1, one, a, a.Off(0, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if i < n {
					if err = w.Off(i, iw-1).Vector().Gemv(Trans, i-1, n-i, one, w.Off(0, iw), a.Off(0, i-1).Vector(), 1, zero, 1); err != nil {
						panic(err)
					}
					if err = w.Off(0, iw-1).Vector().Gemv(NoTrans, i-1, n-i, -one, a.Off(0, i), w.Off(i, iw-1).Vector(), 1, one, 1); err != nil {
						panic(err)
					}
					if err = w.Off(i, iw-1).Vector().Gemv(Trans, i-1, n-i, one, a.Off(0, i), a.Off(0, i-1).Vector(), 1, zero, 1); err != nil {
						panic(err)
					}
					if err = w.Off(0, iw-1).Vector().Gemv(NoTrans, i-1, n-i, -one, w.Off(0, iw), w.Off(i, iw-1).Vector(), 1, one, 1); err != nil {
						panic(err)
					}
				}
				w.Off(0, iw-1).Vector().Scal(i-1, tau.Get(i-1-1), 1)
				alpha = -half * tau.Get(i-1-1) * a.Off(0, i-1).Vector().Dot(i-1, w.Off(0, iw-1).Vector(), 1, 1)
				w.Off(0, iw-1).Vector().Axpy(i-1, alpha, a.Off(0, i-1).Vector(), 1, 1)
			}

		}
	} else {
		//        Reduce first NB columns of lower triangle
		for i = 1; i <= nb; i++ {
			//           Update A(i:n,i)
			if err = a.Off(i-1, i-1).Vector().Gemv(NoTrans, n-i+1, i-1, -one, a.Off(i-1, 0), w.Off(i-1, 0).Vector(), w.Rows, one, 1); err != nil {
				panic(err)
			}
			if err = a.Off(i-1, i-1).Vector().Gemv(NoTrans, n-i+1, i-1, -one, w.Off(i-1, 0), a.Off(i-1, 0).Vector(), a.Rows, one, 1); err != nil {
				panic(err)
			}
			if i < n {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:n,i)
				*a.GetPtr(i, i-1), *tau.GetPtr(i - 1) = Dlarfg(n-i, a.Get(i, i-1), a.Off(min(i+2, n)-1, i-1).Vector(), 1)
				e.Set(i-1, a.Get(i, i-1))
				a.Set(i, i-1, one)

				//              Compute W(i+1:n,i)
				if err = w.Off(i, i-1).Vector().Symv(Lower, n-i, one, a.Off(i, i), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(0, i-1).Vector().Gemv(Trans, n-i, i-1, one, w.Off(i, 0), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(i, i-1).Vector().Gemv(NoTrans, n-i, i-1, -one, a.Off(i, 0), w.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = w.Off(0, i-1).Vector().Gemv(Trans, n-i, i-1, one, a.Off(i, 0), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = w.Off(i, i-1).Vector().Gemv(NoTrans, n-i, i-1, -one, w.Off(i, 0), w.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				w.Off(i, i-1).Vector().Scal(n-i, tau.Get(i-1), 1)
				alpha = -half * tau.Get(i-1) * a.Off(i, i-1).Vector().Dot(n-i, w.Off(i, i-1).Vector(), 1, 1)
				w.Off(i, i-1).Vector().Axpy(n-i, alpha, a.Off(i, i-1).Vector(), 1, 1)
			}

		}
	}
}
