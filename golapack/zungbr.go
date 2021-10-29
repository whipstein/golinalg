package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungbr generates one of the complex unitary matrices Q or P**H
// determined by Zgebrd when reducing a complex matrix A to bidiagonal
// form: A = Q * B * P**H.  Q and P**H are defined as products of
// elementary reflectors H(i) or G(i) respectively.
//
// If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
// is of order M:
// if m >= k, Q = H(1) H(2) . . . H(k) and Zungbr returns the first n
// columns of Q, where m >= n >= k;
// if m < k, Q = H(1) H(2) . . . H(m-1) and Zungbr returns Q as an
// M-by-M matrix.
//
// If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**H
// is of order N:
// if k < n, P**H = G(k) . . . G(2) G(1) and Zungbr returns the first m
// rows of P**H, where n >= m >= k;
// if k >= n, P**H = G(n-1) . . . G(2) G(1) and Zungbr returns P**H as
// an N-by-N matrix.
func Zungbr(vect byte, m, n, k int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery, wantq bool
	var one, zero complex128
	var i, j, lwkopt, mn int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	wantq = vect == 'Q'
	mn = min(m, n)
	lquery = (lwork == -1)
	if !wantq && vect != 'P' {
		err = fmt.Errorf("!wantq && vect != 'P': vect='%c'", vect)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))) {
		err = fmt.Errorf("n < 0 || (wantq && (n > m || n < min(m, k))) || (!wantq && (m > n || m < min(n, k))): vect='%c', m=%v, n=%v, k=%v", vect, m, n, k)
	} else if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, mn) && !lquery {
		err = fmt.Errorf("lwork < max(1, mn) && !lquery: lwork=%v, mn=%v, lquery=%v", lwork, mn, lquery)
	}

	if err == nil {
		work.Set(0, 1)
		if wantq {
			if m >= k {
				if err = Zungqr(m, n, k, a, tau, work, -1); err != nil {
					panic(err)
				}
			} else {
				if m > 1 {
					if err = Zungqr(m-1, m-1, m-1, a.Off(1, 1), tau, work, -1); err != nil {
						panic(err)
					}
				}
			}
		} else {
			if k < n {
				if err = Zunglq(m, n, k, a, tau, work, -1); err != nil {
					panic(err)
				}
			} else {
				if n > 1 {
					if err = Zunglq(n-1, n-1, n-1, a.Off(1, 1), tau, work, -1); err != nil {
						panic(err)
					}
				}
			}
		}
		lwkopt = int(work.GetRe(0))
		lwkopt = max(lwkopt, mn)
	}

	if err != nil {
		gltest.Xerbla2("Zungbr", err)
		return
	} else if lquery {
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		work.Set(0, 1)
		return
	}

	if wantq {
		//        Form Q, determined by a call to Zgebrd to reduce an m-by-k
		//        matrix
		if m >= k {
			//           If m >= k, assume m >= n >= k
			if err = Zungqr(m, n, k, a, tau, work, lwork); err != nil {
				panic(err)
			}

		} else {
			//           If m < k, assume m = n
			//
			//           Shift the vectors which define the elementary reflectors one
			//           column to the right, and set the first row and column of Q
			//           to those of the unit matrix
			for j = m; j >= 2; j-- {
				a.Set(0, j-1, zero)
				for i = j + 1; i <= m; i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1-1))
				}
			}
			a.Set(0, 0, one)
			for i = 2; i <= m; i++ {
				a.Set(i-1, 0, zero)
			}
			if m > 1 {
				//              Form Q(2:m,2:m)
				if err = Zungqr(m-1, m-1, m-1, a.Off(1, 1), tau, work, lwork); err != nil {
					panic(err)
				}
			}
		}
	} else {
		//        Form P**H, determined by a call to Zgebrd to reduce a k-by-n
		//        matrix
		if k < n {
			//           If k < n, assume k <= m <= n
			if err = Zunglq(m, n, k, a, tau, work, lwork); err != nil {
				panic(err)
			}

		} else {
			//           If k >= n, assume m = n
			//
			//           Shift the vectors which define the elementary reflectors one
			//           row downward, and set the first row and column of P**H to
			//           those of the unit matrix
			a.Set(0, 0, one)
			for i = 2; i <= n; i++ {
				a.Set(i-1, 0, zero)
			}
			for j = 2; j <= n; j++ {
				for i = j - 1; i >= 2; i-- {
					a.Set(i-1, j-1, a.Get(i-1-1, j-1))
				}
				a.Set(0, j-1, zero)
			}
			if n > 1 {
				//              Form P**H(2:n,2:n)
				if err = Zunglq(n-1, n-1, n-1, a.Off(1, 1), tau, work, lwork); err != nil {
					panic(err)
				}
			}
		}
	}
	work.SetRe(0, float64(lwkopt))

	return
}
