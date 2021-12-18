package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlagge generates a real general m by n matrix A, by pre- and post-
// multiplying a real diagonal matrix D with random orthogonal matrices:
// A = U*D*V. The lower and upper bandwidths may then be reduced to
// kl and ku by additional orthogonal transformations.
func Dlagge(m, n, kl, ku int, d *mat.Vector, a *mat.Matrix, iseed *[]int, work *mat.Vector) (err error) {
	var one, tau, wa, wb, wn, zero float64
	var i, j int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 || kl > m-1 {
		err = fmt.Errorf("kl < 0 || kl > m-1: kl=%v, m=%v", kl, m)
	} else if ku < 0 || ku > n-1 {
		err = fmt.Errorf("ku < 0 || ku > n-1: ku=%v, n=%v", ku, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("DLAGGE", err)
		return
	}

	//     initialize A to diagonal matrix
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= min(m, n); i++ {
		a.Set(i-1, i-1, d.Get(i-1))
	}

	//     Quick exit if the user wants a diagonal matrix
	if (kl == 0) && (ku == 0) {
		return
	}

	//     pre- and post-multiply A by random orthogonal matrices
	for i = min(m, n); i >= 1; i-- {
		if i < m {
			//
			//           generate random reflection
			//
			golapack.Dlarnv(3, iseed, m-i+1, work)
			wn = work.Nrm2(m-i+1, 1)
			wa = math.Copysign(wn, work.Get(0))
			if wn == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				work.Off(1).Scal(m-i, one/wb, 1)
				work.Set(0, one)
				tau = wb / wa
			}

			//           multiply A(i:m,i:n) by random reflection from the left
			err = work.Off(m).Gemv(Trans, m-i+1, n-i+1, one, a.Off(i-1, i-1), work, 1, zero, 1)
			err = a.Off(i-1, i-1).Ger(m-i+1, n-i+1, -tau, work, 1, work.Off(m), 1)
		}
		if i < n {
			//           generate random reflection
			golapack.Dlarnv(3, iseed, n-i+1, work)
			wn = work.Nrm2(n-i+1, 1)
			wa = math.Copysign(wn, work.Get(0))
			if wn == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				work.Off(1).Scal(n-i, one/wb, 1)
				work.Set(0, one)
				tau = wb / wa
			}

			//           multiply A(i:m,i:n) by random reflection from the right
			if err = work.Off(n).Gemv(NoTrans, m-i+1, n-i+1, one, a.Off(i-1, i-1), work, 1, zero, 1); err != nil {
				panic(err)
			}
			if err = a.Off(i-1, i-1).Ger(m-i+1, n-i+1, -tau, work.Off(n), 1, work, 1); err != nil {
				panic(err)
			}
		}
	}

	//     Reduce number of subdiagonals to KL and number of superdiagonals
	//     to KU
	for i = 1; i <= max(m-1-kl, n-1-ku); i++ {
		if kl <= ku {
			//           annihilate subdiagonal elements first (necessary if KL = 0)
			if i <= min(m-1-kl, n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = a.Off(kl+i-1, i-1).Vector().Nrm2(m-kl-i+1, 1)
				wa = math.Copysign(wn, a.Get(kl+i-1, i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(kl+i-1, i-1) + wa
					a.Off(kl+i, i-1).Vector().Scal(m-kl-i, one/wb, 1)
					a.Set(kl+i-1, i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				if err = work.Gemv(Trans, m-kl-i+1, n-i, one, a.Off(kl+i-1, i), a.Off(kl+i-1, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = a.Off(kl+i-1, i).Ger(m-kl-i+1, n-i, -tau, a.Off(kl+i-1, i-1).Vector(), 1, work, 1); err != nil {
					panic(err)
				}
				a.Set(kl+i-1, i-1, -wa)
			}

			if i <= min(n-1-ku, m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = a.Off(i-1, ku+i-1).Vector().Nrm2(n-ku-i+1, a.Rows)
				wa = math.Copysign(wn, a.Get(i-1, ku+i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, ku+i-1) + wa
					a.Off(i-1, ku+i).Vector().Scal(n-ku-i, one/wb, a.Rows)
					a.Set(i-1, ku+i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				if err = work.Gemv(NoTrans, m-i, n-ku-i+1, one, a.Off(i, ku+i-1), a.Off(i-1, ku+i-1).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = a.Off(i, ku+i-1).Ger(m-i, n-ku-i+1, -tau, work, 1, a.Off(i-1, ku+i-1).Vector(), a.Rows); err != nil {
					panic(err)
				}
				a.Set(i-1, ku+i-1, -wa)
			}
		} else {
			//           annihilate superdiagonal elements first (necessary if
			//           KU = 0)
			if i <= min(n-1-ku, m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = a.Off(i-1, ku+i-1).Vector().Nrm2(n-ku-i+1, a.Rows)
				wa = math.Copysign(wn, a.Get(i-1, ku+i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, ku+i-1) + wa
					a.Off(i-1, ku+i).Vector().Scal(n-ku-i, one/wb, a.Rows)
					a.Set(i-1, ku+i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				if err = work.Gemv(NoTrans, m-i, n-ku-i+1, one, a.Off(i, ku+i-1), a.Off(i-1, ku+i-1).Vector(), a.Rows, zero, 1); err != nil {
					panic(err)
				}
				if err = a.Off(i, ku+i-1).Ger(m-i, n-ku-i+1, -tau, work, 1, a.Off(i-1, ku+i-1).Vector(), a.Rows); err != nil {
					panic(err)
				}
				a.Set(i-1, ku+i-1, -wa)
			}

			if i <= min(m-1-kl, n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = a.Off(kl+i-1, i-1).Vector().Nrm2(m-kl-i+1, 1)
				wa = math.Copysign(wn, a.Get(kl+i-1, i-1))
				if wn == zero {
					tau = zero
				} else {
					wb = a.Get(kl+i-1, i-1) + wa
					a.Off(kl+i, i-1).Vector().Scal(m-kl-i, one/wb, 1)
					a.Set(kl+i-1, i-1, one)
					tau = wb / wa
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				if err = work.Gemv(mat.Trans, m-kl-i+1, n-i, one, a.Off(kl+i-1, i), a.Off(kl+i-1, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}
				if err = a.Off(kl+i-1, i).Ger(m-kl-i+1, n-i, -tau, a.Off(kl+i-1, i-1).Vector(), 1, work, 1); err != nil {
					panic(err)
				}
				a.Set(kl+i-1, i-1, -wa)
			}
		}

		if i <= n {
			for j = kl + i + 1; j <= m; j++ {
				a.Set(j-1, i-1, zero)
			}
		}

		if i <= m {
			for j = ku + i + 1; j <= n; j++ {
				a.Set(i-1, j-1, zero)
			}
		}
	}

	return
}
