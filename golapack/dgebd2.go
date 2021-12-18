package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebd2 reduces a real general m by n matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Dgebd2(m, n int, a *mat.Matrix, d, e, tauq, taup, work *mat.Vector) (err error) {
	var one, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgebd2", err)
		return
	}

	if m >= n {
		//        Reduce to upper bidiagonal form
		for i = 1; i <= n; i++ {
			//           Generate elementary reflector H(i) to annihilate A(i+1:m,i)
			*a.GetPtr(i-1, i-1), *tauq.GetPtr(i - 1) = Dlarfg(m-i+1, a.Get(i-1, i-1), a.Off(min(i+1, m)-1, i-1).Vector(), 1)
			d.Set(i-1, a.Get(i-1, i-1))
			a.Set(i-1, i-1, one)

			//           Apply H(i) to A(i:m,i+1:n) from the left
			if i < n {
				Dlarf(Left, m-i+1, n-i, a.Off(i-1, i-1).Vector(), 1, tauq.Get(i-1), a.Off(i-1, i), work)
			}
			a.Set(i-1, i-1, d.Get(i-1))

			if i < n {
				//              Generate elementary reflector G(i) to annihilate
				//              A(i,i+2:n)
				*a.GetPtr(i-1, i), *taup.GetPtr(i - 1) = Dlarfg(n-i, a.Get(i-1, i), a.Off(i-1, min(i+2, n)-1).Vector(), a.Rows)
				e.Set(i-1, a.Get(i-1, i))
				a.Set(i-1, i, one)

				//              Apply G(i) to A(i+1:m,i+1:n) from the right
				Dlarf(Right, m-i, n-i, a.Off(i-1, i).Vector(), a.Rows, taup.Get(i-1), a.Off(i, i), work)
				a.Set(i-1, i, e.Get(i-1))
			} else {
				taup.Set(i-1, zero)
			}
		}
	} else {
		//        Reduce to lower bidiagonal form
		for i = 1; i <= m; i++ {
			//           Generate elementary reflector G(i) to annihilate A(i,i+1:n)
			*a.GetPtr(i-1, i-1), *taup.GetPtr(i - 1) = Dlarfg(n-i+1, a.Get(i-1, i-1), a.Off(i-1, min(i+1, n)-1).Vector(), a.Rows)
			d.Set(i-1, a.Get(i-1, i-1))
			a.Set(i-1, i-1, one)

			//           Apply G(i) to A(i+1:m,i:n) from the right
			if i < m {
				Dlarf(Right, m-i, n-i+1, a.Off(i-1, i-1).Vector(), a.Rows, taup.Get(i-1), a.Off(i, i-1), work)
			}
			a.Set(i-1, i-1, d.Get(i-1))

			if i < m {
				//              Generate elementary reflector H(i) to annihilate
				//              A(i+2:m,i)
				*a.GetPtr(i, i-1), *tauq.GetPtr(i - 1) = Dlarfg(m-i, a.Get(i, i-1), a.Off(min(i+2, m)-1, i-1).Vector(), 1)
				e.Set(i-1, a.Get(i, i-1))
				a.Set(i, i-1, one)

				//              Apply H(i) to A(i+1:m,i+1:n) from the left
				Dlarf(Left, m-i, n-i, a.Off(i, i-1).Vector(), 1, tauq.Get(i-1), a.Off(i, i), work)
				a.Set(i, i-1, e.Get(i-1))
			} else {
				tauq.Set(i-1, zero)
			}
		}
	}

	return
}
