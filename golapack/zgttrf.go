package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgttrf computes an LU factorization of a complex tridiagonal matrix A
// using elimination with partial pivoting and row interchanges.
//
// The factorization has the form
//    A = L * U
// where L is a product of permutation and unit lower bidiagonal
// matrices and U is upper triangular with nonzeros in only the main
// diagonal and first two superdiagonals.
func Zgttrf(n int, dl, d, du, du2 *mat.CVector, ipiv *[]int) (info int, err error) {
	var fact, temp complex128
	var zero float64
	var i int

	zero = 0.0

	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
		gltest.Xerbla2("Zgttrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Initialize IPIV(i) = i and DU2(i) = 0
	for i = 1; i <= n; i++ {
		(*ipiv)[i-1] = i
	}
	for i = 1; i <= n-2; i++ {
		du2.SetRe(i-1, zero)
	}

	for i = 1; i <= n-2; i++ {
		if cabs1(d.Get(i-1)) >= cabs1(dl.Get(i-1)) {
			//           No row interchange required, eliminate DL(I)
			if cabs1(d.Get(i-1)) != zero {
				fact = dl.Get(i-1) / d.Get(i-1)
				dl.Set(i-1, fact)
				d.Set(i, d.Get(i)-fact*du.Get(i-1))
			}
		} else {
			//           Interchange rows I and I+1, eliminate DL(I)
			fact = d.Get(i-1) / dl.Get(i-1)
			d.Set(i-1, dl.Get(i-1))
			dl.Set(i-1, fact)
			temp = du.Get(i - 1)
			du.Set(i-1, d.Get(i))
			d.Set(i, temp-fact*d.Get(i))
			du2.Set(i-1, du.Get(i))
			du.Set(i, -fact*du.Get(i))
			(*ipiv)[i-1] = i + 1
		}
	}
	if n > 1 {
		i = n - 1
		if cabs1(d.Get(i-1)) >= cabs1(dl.Get(i-1)) {
			if cabs1(d.Get(i-1)) != zero {
				fact = dl.Get(i-1) / d.Get(i-1)
				dl.Set(i-1, fact)
				d.Set(i, d.Get(i)-fact*du.Get(i-1))
			}
		} else {
			fact = d.Get(i-1) / dl.Get(i-1)
			d.Set(i-1, dl.Get(i-1))
			dl.Set(i-1, fact)
			temp = du.Get(i - 1)
			du.Set(i-1, d.Get(i))
			d.Set(i, temp-fact*d.Get(i))
			(*ipiv)[i-1] = i + 1
		}
	}

	//     Check for a zero on the diagonal of U.
	for i = 1; i <= n; i++ {
		if cabs1(d.Get(i-1)) == zero {
			info = i
			return
		}
	}

	return
}
