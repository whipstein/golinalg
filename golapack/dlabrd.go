package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dlabrd reduces the first NB rows and columns of a real general
// m by n matrix A to upper or lower bidiagonal form by an orthogonal
// transformation Q**T * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.
//
// If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
// bidiagonal form.
//
// This is an auxiliary routine called by DGEBRD
func Dlabrd(m, n, nb int, a *mat.Matrix, d, e, tauq, taup *mat.Vector, x, y *mat.Matrix) {
	var one, zero float64
	var i int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	if m >= n {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= nb; i++ {
			//           Update A(i:m,i)
			if err = a.Off(i-1, i-1).Vector().Gemv(NoTrans, m-i+1, i-1, -one, a.Off(i-1, 0), y.Off(i-1, 0).Vector(), y.Rows, one, 1); err != nil {
				panic(err)
			}
			if err = a.Off(i-1, i-1).Vector().Gemv(NoTrans, m-i+1, i-1, -one, x.Off(i-1, 0), a.Off(0, i-1).Vector(), 1, one, 1); err != nil {
				panic(err)
			}

			//           Generate reflection Q(i) to annihilate A(i+1:m,i)
			*a.GetPtr(i-1, i-1), *tauq.GetPtr(i - 1) = Dlarfg(m-i+1, a.Get(i-1, i-1), a.Off(min(i+1, m)-1, i-1).Vector(), 1)
			d.Set(i-1, a.Get(i-1, i-1))
			if i < n {
				a.Set(i-1, i-1, one)

				//              Compute Y(i+1:n,i)
				if err = y.Off(i, i-1).Vector().Gemv(Trans, m-i+1, n-i, one, a.Off(i-1, i), a.Off(i-1, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(0, i-1).Vector().Gemv(Trans, m-i+1, i-1, one, a.Off(i-1, 0), a.Off(i-1, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(i, i-1).Vector().Gemv(NoTrans, n-i, i-1, -one, y.Off(i, 0), y.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = y.Off(0, i-1).Vector().Gemv(Trans, m-i+1, i-1, one, x.Off(i-1, 0), a.Off(i-1, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(i, i-1).Vector().Gemv(Trans, i-1, n-i, -one, a.Off(0, i), y.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				y.Off(i, i-1).Vector().Scal(n-i, tauq.Get(i-1), 1)

				//              Update A(i,i+1:n)
				if err = a.Off(i-1, i).Vector().Gemv(NoTrans, n-i, i, -one, y.Off(i, 0), a.Off(i-1, 0).Vector(), a.Rows, one, a.Rows); err != nil {
					panic(err)
				}
				if err = a.Off(i-1, i).Vector().Gemv(Trans, i-1, n-i, -one, a.Off(0, i), x.Off(i-1, 0).Vector(), x.Rows, one, a.Rows); err != nil {
					panic(err)
				}

				//              Generate reflection P(i) to annihilate A(i,i+2:n)
				*a.GetPtr(i-1, i), *taup.GetPtr(i - 1) = Dlarfg(n-i, a.Get(i-1, i), a.Off(i-1, min(i+2, n)-1).Vector(), a.Rows)
				e.Set(i-1, a.Get(i-1, i))
				a.Set(i-1, i, one)

				//              Compute X(i+1:m,i)
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, n-i, one, a.Off(i, i), a.Off(i-1, i).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(0, i-1).Vector().Gemv(Trans, n-i, i, one, y.Off(i, 0), a.Off(i-1, i).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i, -one, a.Off(i, 0), x.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = x.Off(0, i-1).Vector().Gemv(NoTrans, i-1, n-i, one, a.Off(0, i), a.Off(i-1, i).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i-1, -one, x.Off(i, 0), x.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				x.Off(i, i-1).Vector().Scal(m-i, taup.Get(i-1), 1)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= nb; i++ {
			//           Update A(i,i:n)
			if err = a.Off(i-1, i-1).Vector().Gemv(NoTrans, n-i+1, i-1, -one, y.Off(i-1, 0), a.Off(i-1, 0).Vector(), a.Rows, one, a.Rows); err != nil {
				panic(err)
			}
			if err = a.Off(i-1, i-1).Vector().Gemv(Trans, i-1, n-i+1, -one, a.Off(0, i-1), x.Off(i-1, 0).Vector(), x.Rows, one, a.Rows); err != nil {
				panic(err)
			}

			//           Generate reflection P(i) to annihilate A(i,i+1:n)
			*a.GetPtr(i-1, i-1), *taup.GetPtr(i - 1) = Dlarfg(n-i+1, a.Get(i-1, i-1), a.Off(i-1, min(i+1, n)-1).Vector(), a.Rows)
			d.Set(i-1, a.Get(i-1, i-1))
			if i < m {
				a.Set(i-1, i-1, one)

				//              Compute X(i+1:m,i)
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, n-i+1, one, a.Off(i, i-1), a.Off(i-1, i-1).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(0, i-1).Vector().Gemv(Trans, n-i+1, i-1, one, y.Off(i-1, 0), a.Off(i-1, i-1).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i-1, -one, a.Off(i, 0), x.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = x.Off(0, i-1).Vector().Gemv(NoTrans, i-1, n-i+1, one, a.Off(0, i-1), a.Off(i-1, i-1).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = x.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i-1, -one, x.Off(i, 0), x.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				x.Off(i, i-1).Vector().Scal(m-i, taup.Get(i-1), 1)

				//              Update A(i+1:m,i)
				if err = a.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i-1, -one, a.Off(i, 0), y.Off(i-1, 0).Vector(), y.Rows, one, 1); err != nil {
					panic(err)
				}
				if err = a.Off(i, i-1).Vector().Gemv(NoTrans, m-i, i, -one, x.Off(i, 0), a.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}

				//              Generate reflection Q(i) to annihilate A(i+2:m,i)
				*a.GetPtr(i, i-1), *tauq.GetPtr(i - 1) = Dlarfg(m-i, a.Get(i, i-1), a.Off(min(i+2, m)-1, i-1).Vector(), 1)
				e.Set(i-1, a.Get(i, i-1))
				a.Set(i, i-1, one)

				//              Compute Y(i+1:n,i)
				if err = y.Off(i, i-1).Vector().Gemv(Trans, m-i, n-i, one, a.Off(i, i), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(0, i-1).Vector().Gemv(Trans, m-i, i-1, one, a.Off(i, 0), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(i, i-1).Vector().Gemv(NoTrans, n-i, i-1, -one, y.Off(i, 0), y.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				if err = y.Off(0, i-1).Vector().Gemv(Trans, m-i, i, one, x.Off(i, 0), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = y.Off(i, i-1).Vector().Gemv(Trans, i, n-i, -one, a.Off(0, i), y.Off(0, i-1).Vector(), 1, one, 1); err != nil {
					panic(err)
				}
				y.Off(i, i-1).Vector().Scal(n-i, tauq.Get(i-1), 1)
			}
		}
	}
}
