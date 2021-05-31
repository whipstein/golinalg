package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgttrf computes an LU factorization of a real tridiagonal matrix A
// using elimination with partial pivoting and row interchanges.
//
// The factorization has the form
//    A = L * U
// where L is a product of permutation and unit lower bidiagonal
// matrices and U is upper triangular with nonzeros in only the main
// diagonal and first two superdiagonals.
func Dgttrf(n *int, dl, d, du, du2 *mat.Vector, ipiv *[]int, info *int) {
	var fact, temp, zero float64
	var i int

	zero = 0.0

	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("DGTTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Initialize IPIV(i) = i and DU2(I) = 0
	for i = 1; i <= (*n); i++ {
		(*ipiv)[i-1] = i
	}
	for i = 1; i <= (*n)-2; i++ {
		du2.Set(i-1, zero)
	}

	for i = 1; i <= (*n)-2; i++ {
		if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
			//           No row interchange required, eliminate DL(I)
			if d.Get(i-1) != zero {
				fact = dl.Get(i-1) / d.Get(i-1)
				dl.Set(i-1, fact)
				d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
			}
		} else {
			//           Interchange rows I and I+1, eliminate DL(I)
			fact = d.Get(i-1) / dl.Get(i-1)
			d.Set(i-1, dl.Get(i-1))
			dl.Set(i-1, fact)
			temp = du.Get(i - 1)
			du.Set(i-1, d.Get(i+1-1))
			d.Set(i+1-1, temp-fact*d.Get(i+1-1))
			du2.Set(i-1, du.Get(i+1-1))
			du.Set(i+1-1, -fact*du.Get(i+1-1))
			(*ipiv)[i-1] = i + 1
		}
	}
	if (*n) > 1 {
		i = (*n) - 1
		if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
			if d.Get(i-1) != zero {
				fact = dl.Get(i-1) / d.Get(i-1)
				dl.Set(i-1, fact)
				d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
			}
		} else {
			fact = d.Get(i-1) / dl.Get(i-1)
			d.Set(i-1, dl.Get(i-1))
			dl.Set(i-1, fact)
			temp = du.Get(i - 1)
			du.Set(i-1, d.Get(i+1-1))
			d.Set(i+1-1, temp-fact*d.Get(i+1-1))
			(*ipiv)[i-1] = i + 1
		}
	}

	//     Check for a zero on the diagonal of U.
	for i = 1; i <= (*n); i++ {
		if d.Get(i-1) == zero {
			(*info) = i
			goto label50
		}
	}
label50:
}
