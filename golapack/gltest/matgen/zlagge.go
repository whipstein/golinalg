package matgen

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlagge generates a complex general m by n matrix A, by pre- and post-
// multiplying a real diagonal matrix D with random unitary matrices:
// A = U*D*V. The lower and upper bandwidths may then be reduced to
// kl and ku by additional unitary transformations.
func Zlagge(m, n, kl, ku int, d *mat.Vector, a *mat.CMatrix, iseed *[]int, work *mat.CVector) (err error) {
	var one, tau, wa, wb, zero complex128
	var wn float64
	var i, j int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zlagge", err)
		return
	}

	//     initialize A to diagonal matrix
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for i = 1; i <= min(m, n); i++ {
		a.Set(i-1, i-1, complex(d.Get(i-1), 0))
	}

	//     Quick exit if the user wants a diagonal matrix
	if (kl == 0) && (ku == 0) {
		return
	}

	//     pre- and post-multiply A by random unitary matrices
	for i = min(m, n); i >= 1; i-- {
		if i < m {
			//           generate random reflection
			golapack.Zlarnv(3, iseed, m-i+1, work)
			wn = goblas.Dznrm2(m-i+1, work.Off(0, 1))
			wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
			if complex(wn, 0) == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				goblas.Zscal(m-i, one/wb, work.Off(1, 1))
				work.Set(0, one)
				tau = wb / wa
			}

			//           multiply A(i:m,i:n) by random reflection from the left
			if err = goblas.Zgemv(ConjTrans, m-i+1, n-i+1, one, a.Off(i-1, i-1), work.Off(0, 1), zero, work.Off(m, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(m-i+1, n-i+1, -tau, work.Off(0, 1), work.Off(m, 1), a.Off(i-1, i-1)); err != nil {
				panic(err)
			}
		}
		if i < n {
			//           generate random reflection
			golapack.Zlarnv(3, iseed, n-i+1, work)
			wn = goblas.Dznrm2(n-i+1, work.Off(0, 1))
			wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
			if complex(wn, 0) == zero {
				tau = zero
			} else {
				wb = work.Get(0) + wa
				goblas.Zscal(n-i, one/wb, work.Off(1, 1))
				work.Set(0, one)
				tau = complex(real(wb/wa), 0)
			}

			//           multiply A(i:m,i:n) by random reflection from the right
			if err = goblas.Zgemv(NoTrans, m-i+1, n-i+1, one, a.Off(i-1, i-1), work.Off(0, 1), zero, work.Off(n, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(m-i+1, n-i+1, -tau, work.Off(n, 1), work.Off(0, 1), a.Off(i-1, i-1)); err != nil {
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
				wn = goblas.Dznrm2(m-kl-i+1, a.CVector(kl+i-1, i-1, 1))
				wa = complex(wn/a.GetMag(kl+i-1, i-1), 0) * a.Get(kl+i-1, i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(kl+i-1, i-1) + wa
					goblas.Zscal(m-kl-i, one/wb, a.CVector(kl+i, i-1, 1))
					a.Set(kl+i-1, i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				if err = goblas.Zgemv(ConjTrans, m-kl-i+1, n-i, one, a.Off(kl+i-1, i), a.CVector(kl+i-1, i-1, 1), zero, work.Off(0, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgerc(m-kl-i+1, n-i, -tau, a.CVector(kl+i-1, i-1, 1), work.Off(0, 1), a.Off(kl+i-1, i)); err != nil {
					panic(err)
				}
				a.Set(kl+i-1, i-1, -wa)
			}

			if i <= min(n-1-ku, m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dznrm2(n-ku-i+1, a.CVector(i-1, ku+i-1))
				wa = complex(wn/a.GetMag(i-1, ku+i-1), 0) * a.Get(i-1, ku+i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, ku+i-1) + wa
					goblas.Zscal(n-ku-i, one/wb, a.CVector(i-1, ku+i))
					a.Set(i-1, ku+i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				golapack.Zlacgv(n-ku-i+1, a.CVector(i-1, ku+i-1))
				if err = goblas.Zgemv(NoTrans, m-i, n-ku-i+1, one, a.Off(i, ku+i-1), a.CVector(i-1, ku+i-1), zero, work.Off(0, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgerc(m-i, n-ku-i+1, -tau, work.Off(0, 1), a.CVector(i-1, ku+i-1), a.Off(i, ku+i-1)); err != nil {
					panic(err)
				}
				a.Set(i-1, ku+i-1, -wa)
			}
		} else {
			//           annihilate superdiagonal elements first (necessary if
			//           KU = 0)
			if i <= min(n-1-ku, m) {
				//              generate reflection to annihilate A(i,ku+i+1:n)
				wn = goblas.Dznrm2(n-ku-i+1, a.CVector(i-1, ku+i-1))
				wa = complex(wn/a.GetMag(i-1, ku+i-1), 0) * a.Get(i-1, ku+i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(i-1, ku+i-1) + wa
					goblas.Zscal(n-ku-i, one/wb, a.CVector(i-1, ku+i))
					a.Set(i-1, ku+i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(i+1:m,ku+i:n) from the right
				golapack.Zlacgv(n-ku-i+1, a.CVector(i-1, ku+i-1))
				if err = goblas.Zgemv(NoTrans, m-i, n-ku-i+1, one, a.Off(i, ku+i-1), a.CVector(i-1, ku+i-1), zero, work.Off(0, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgerc(m-i, n-ku-i+1, -tau, work.Off(0, 1), a.CVector(i-1, ku+i-1), a.Off(i, ku+i-1)); err != nil {
					panic(err)
				}
				a.Set(i-1, ku+i-1, -wa)
			}

			if i <= min(m-1-kl, n) {
				//              generate reflection to annihilate A(kl+i+1:m,i)
				wn = goblas.Dznrm2(m-kl-i+1, a.CVector(kl+i-1, i-1, 1))
				wa = complex(wn/a.GetMag(kl+i-1, i-1), 0) * a.Get(kl+i-1, i-1)
				if complex(wn, 0) == zero {
					tau = zero
				} else {
					wb = a.Get(kl+i-1, i-1) + wa
					goblas.Zscal(m-kl-i, one/wb, a.CVector(kl+i, i-1, 1))
					a.Set(kl+i-1, i-1, one)
					tau = complex(real(wb/wa), 0)
				}

				//              apply reflection to A(kl+i:m,i+1:n) from the left
				if err = goblas.Zgemv(ConjTrans, m-kl-i+1, n-i, one, a.Off(kl+i-1, i), a.CVector(kl+i-1, i-1, 1), zero, work.Off(0, 1)); err != nil {
					panic(err)
				}
				if err = goblas.Zgerc(m-kl-i+1, n-i, -tau, a.CVector(kl+i-1, i-1, 1), work.Off(0, 1), a.Off(kl+i-1, i)); err != nil {
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
