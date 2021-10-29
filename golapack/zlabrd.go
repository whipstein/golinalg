package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlabrd reduces the first NB rows and columns of a complex general
// m by n matrix A to upper or lower real bidiagonal form by a unitary
// transformation Q**H * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.
//
// If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
// bidiagonal form.
//
// This is an auxiliary routine called by ZGEBRD
func Zlabrd(m, n, nb int, a *mat.CMatrix, d, e *mat.Vector, tauq, taup *mat.CVector, x, y *mat.CMatrix) {
	var alpha, one, zero complex128
	var i int
	var err error

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	if m >= n {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= nb; i++ {
			//           Update A(i:m,i)
			Zlacgv(i-1, y.CVector(i-1, 0))
			if err = goblas.Zgemv(NoTrans, m-i+1, i-1, -one, a.Off(i-1, 0), y.CVector(i-1, 0, y.Rows), one, a.CVector(i-1, i-1, 1)); err != nil {
				panic(err)
			}
			Zlacgv(i-1, y.CVector(i-1, 0))
			if err = goblas.Zgemv(NoTrans, m-i+1, i-1, -one, x.Off(i-1, 0), a.CVector(0, i-1, 1), one, a.CVector(i-1, i-1, 1)); err != nil {
				panic(err)
			}

			//           Generate reflection Q(i) to annihilate A(i+1:m,i)
			alpha = a.Get(i-1, i-1)
			alpha, *tauq.GetPtr(i - 1) = Zlarfg(m-i+1, alpha, a.CVector(min(i+1, m)-1, i-1, 1))
			d.Set(i-1, real(alpha))
			if i < n {
				a.Set(i-1, i-1, one)

				//              Compute Y(i+1:n,i)
				if err = goblas.Zgemv(ConjTrans, m-i+1, n-i, one, a.Off(i-1, i), a.CVector(i-1, i-1, 1), zero, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, m-i+1, i-1, one, a.Off(i-1, 0), a.CVector(i-1, i-1, 1), zero, y.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, n-i, i-1, -one, y.Off(i, 0), y.CVector(0, i-1, 1), one, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, m-i+1, i-1, one, x.Off(i-1, 0), a.CVector(i-1, i-1, 1), zero, y.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, i-1, n-i, -one, a.Off(0, i), y.CVector(0, i-1, 1), one, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				goblas.Zscal(n-i, tauq.Get(i-1), y.CVector(i, i-1, 1))

				//              Update A(i,i+1:n)
				Zlacgv(n-i, a.CVector(i-1, i))
				Zlacgv(i, a.CVector(i-1, 0))
				if err = goblas.Zgemv(NoTrans, n-i, i, -one, y.Off(i, 0), a.CVector(i-1, 0, a.Rows), one, a.CVector(i-1, i, a.Rows)); err != nil {
					panic(err)
				}
				Zlacgv(i, a.CVector(i-1, 0))
				Zlacgv(i-1, x.CVector(i-1, 0))
				if err = goblas.Zgemv(ConjTrans, i-1, n-i, -one, a.Off(0, i), x.CVector(i-1, 0, x.Rows), one, a.CVector(i-1, i, a.Rows)); err != nil {
					panic(err)
				}
				Zlacgv(i-1, x.CVector(i-1, 0))

				//              Generate reflection P(i) to annihilate A(i,i+2:n)
				alpha = a.Get(i-1, i)
				alpha, *taup.GetPtr(i - 1) = Zlarfg(n-i, alpha, a.CVector(i-1, min(i+2, n)-1))
				e.Set(i-1, real(alpha))
				a.Set(i-1, i, one)

				//              Compute X(i+1:m,i)
				if err = goblas.Zgemv(NoTrans, m-i, n-i, one, a.Off(i, i), a.CVector(i-1, i, a.Rows), zero, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, n-i, i, one, y.Off(i, 0), a.CVector(i-1, i, a.Rows), zero, x.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, m-i, i, -one, a.Off(i, 0), x.CVector(0, i-1, 1), one, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, i-1, n-i, one, a.Off(0, i), a.CVector(i-1, i, a.Rows), zero, x.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, m-i, i-1, -one, x.Off(i, 0), x.CVector(0, i-1, 1), one, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				goblas.Zscal(m-i, taup.Get(i-1), x.CVector(i, i-1, 1))
				Zlacgv(n-i, a.CVector(i-1, i))
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= nb; i++ {
			//           Update A(i,i:n)
			Zlacgv(n-i+1, a.CVector(i-1, i-1))
			Zlacgv(i-1, a.CVector(i-1, 0))
			if err = goblas.Zgemv(NoTrans, n-i+1, i-1, -one, y.Off(i-1, 0), a.CVector(i-1, 0, a.Rows), one, a.CVector(i-1, i-1, a.Rows)); err != nil {
				panic(err)
			}
			Zlacgv(i-1, a.CVector(i-1, 0))
			Zlacgv(i-1, x.CVector(i-1, 0))
			if err = goblas.Zgemv(ConjTrans, i-1, n-i+1, -one, a.Off(0, i-1), x.CVector(i-1, 0, x.Rows), one, a.CVector(i-1, i-1, a.Rows)); err != nil {
				panic(err)
			}
			Zlacgv(i-1, x.CVector(i-1, 0))

			//           Generate reflection P(i) to annihilate A(i,i+1:n)
			alpha = a.Get(i-1, i-1)
			alpha, *taup.GetPtr(i - 1) = Zlarfg(n-i+1, alpha, a.CVector(i-1, min(i+1, n)-1))
			d.Set(i-1, real(alpha))
			if i < m {
				a.Set(i-1, i-1, one)

				//              Compute X(i+1:m,i)
				if err = goblas.Zgemv(NoTrans, m-i, n-i+1, one, a.Off(i, i-1), a.CVector(i-1, i-1, a.Rows), zero, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, n-i+1, i-1, one, y.Off(i-1, 0), a.CVector(i-1, i-1, a.Rows), zero, x.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, m-i, i-1, -one, a.Off(i, 0), x.CVector(0, i-1, 1), one, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, i-1, n-i+1, one, a.Off(0, i-1), a.CVector(i-1, i-1, a.Rows), zero, x.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, m-i, i-1, -one, x.Off(i, 0), x.CVector(0, i-1, 1), one, x.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				goblas.Zscal(m-i, taup.Get(i-1), x.CVector(i, i-1, 1))
				Zlacgv(n-i+1, a.CVector(i-1, i-1))

				//              Update A(i+1:m,i)
				Zlacgv(i-1, y.CVector(i-1, 0))
				if err = goblas.Zgemv(NoTrans, m-i, i-1, -one, a.Off(i, 0), y.CVector(i-1, 0, y.Rows), one, a.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				Zlacgv(i-1, y.CVector(i-1, 0))
				if err = goblas.Zgemv(NoTrans, m-i, i, -one, x.Off(i, 0), a.CVector(0, i-1, 1), one, a.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}

				//              Generate reflection Q(i) to annihilate A(i+2:m,i)
				alpha = a.Get(i, i-1)
				alpha, *tauq.GetPtr(i - 1) = Zlarfg(m-i, alpha, a.CVector(min(i+2, m)-1, i-1, 1))
				e.Set(i-1, real(alpha))
				a.Set(i, i-1, one)

				//              Compute Y(i+1:n,i)
				if err = goblas.Zgemv(ConjTrans, m-i, n-i, one, a.Off(i, i), a.CVector(i, i-1, 1), zero, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, m-i, i-1, one, a.Off(i, 0), a.CVector(i, i-1, 1), zero, y.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(NoTrans, n-i, i-1, -one, y.Off(i, 0), y.CVector(0, i-1, 1), one, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, m-i, i, one, x.Off(i, 0), a.CVector(i, i-1, 1), zero, y.CVector(0, i-1, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgemv(ConjTrans, i, n-i, -one, a.Off(0, i), y.CVector(0, i-1, 1), one, y.CVector(i, i-1, 1)); err != nil {
					panic(err)
				}
				goblas.Zscal(n-i, tauq.Get(i-1), y.CVector(i, i-1, 1))
			} else {
				Zlacgv(n-i+1, a.CVector(i-1, i-1))
			}
		}
	}
}
