package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgebd2 reduces a complex general m by n matrix A to upper or lower
// real bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Zgebd2(m, n int, a *mat.CMatrix, d, e *mat.Vector, tauq, taup, work *mat.CVector) (err error) {
	var alpha, one, zero complex128
	var i int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zgebd2", err)
		return
	}

	if m >= n {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= n; i++ {
			//           Generate elementary reflector H(i) to annihilate A(i+1:m,i)
			alpha = a.Get(i-1, i-1)
			alpha, *tauq.GetPtr(i - 1) = Zlarfg(m-i+1, alpha, a.CVector(min(i+1, m)-1, i-1, 1))
			d.Set(i-1, real(alpha))
			a.Set(i-1, i-1, one)

			//           Apply H(i)**H to A(i:m,i+1:n) from the left
			if i < n {
				Zlarf(Left, m-i+1, n-i, a.CVector(i-1, i-1, 1), tauq.GetConj(i-1), a.Off(i-1, i), work)
			}
			a.SetRe(i-1, i-1, d.Get(i-1))

			if i < n {
				//              Generate elementary reflector G(i) to annihilate
				//              A(i,i+2:n)
				Zlacgv(n-i, a.CVector(i-1, i))
				alpha = a.Get(i-1, i)
				alpha, *taup.GetPtr(i - 1) = Zlarfg(n-i, alpha, a.CVector(i-1, min(i+2, n)-1))
				e.Set(i-1, real(alpha))
				a.Set(i-1, i, one)

				//              Apply G(i) to A(i+1:m,i+1:n) from the right
				Zlarf(Right, m-i, n-i, a.CVector(i-1, i), taup.Get(i-1), a.Off(i, i), work)
				Zlacgv(n-i, a.CVector(i-1, i))
				a.SetRe(i-1, i, e.Get(i-1))
			} else {
				taup.Set(i-1, zero)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= m; i++ {
			//           Generate elementary reflector G(i) to annihilate A(i,i+1:n)
			Zlacgv(n-i+1, a.CVector(i-1, i-1))
			alpha = a.Get(i-1, i-1)
			alpha, *taup.GetPtr(i - 1) = Zlarfg(n-i+1, alpha, a.CVector(i-1, min(i+1, n)-1))
			d.Set(i-1, real(alpha))
			a.Set(i-1, i-1, one)

			//           Apply G(i) to A(i+1:m,i:n) from the right
			if i < m {
				Zlarf(Right, m-i, n-i+1, a.CVector(i-1, i-1), taup.Get(i-1), a.Off(i, i-1), work)
			}
			Zlacgv(n-i+1, a.CVector(i-1, i-1))
			a.SetRe(i-1, i-1, d.Get(i-1))

			if i < m {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:m,i)
				alpha = a.Get(i, i-1)
				alpha, *tauq.GetPtr(i - 1) = Zlarfg(m-i, alpha, a.CVector(min(i+2, m)-1, i-1, 1))
				e.Set(i-1, real(alpha))
				a.Set(i, i-1, one)

				//              Apply H(i)**H to A(i+1:m,i+1:n) from the left
				Zlarf(Left, m-i, n-i, a.CVector(i, i-1, 1), tauq.GetConj(i-1), a.Off(i, i), work)
				a.SetRe(i, i-1, e.Get(i-1))
			} else {
				tauq.Set(i-1, zero)
			}
		}
	}

	return
}
