package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgtsv solves the equation
//
//    A*X = B,
//
// where A is an N-by-N tridiagonal matrix, by Gaussian elimination with
// partial pivoting.
//
// Note that the equation  A**T *X = B  may be solved by interchanging the
// order of the arguments DU and DL.
func Zgtsv(n, nrhs *int, dl, d, du *mat.CVector, b *mat.CMatrix, ldb, info *int) {
	var mult, temp, zero complex128
	var j, k int

	zero = (0.0 + 0.0*1i)

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGTSV "), -(*info))
		return
	}

	if (*n) == 0 {
		return
	}

	for k = 1; k <= (*n)-1; k++ {
		if dl.Get(k-1) == zero {
			//           Subdiagonal is zero, no elimination is required.
			if d.Get(k-1) == zero {
				//              Diagonal is zero: set INFO = K and return; a unique
				//              solution can not be found.
				(*info) = k
				return
			}
		} else if Cabs1(d.Get(k-1)) >= Cabs1(dl.Get(k-1)) {
			//           No row interchange required
			mult = dl.Get(k-1) / d.Get(k-1)
			d.Set(k+1-1, d.Get(k+1-1)-mult*du.Get(k-1))
			for j = 1; j <= (*nrhs); j++ {
				b.Set(k+1-1, j-1, b.Get(k+1-1, j-1)-mult*b.Get(k-1, j-1))
			}
			if k < ((*n) - 1) {
				dl.Set(k-1, zero)
			}
		} else {
			//           Interchange rows K and K+1
			mult = d.Get(k-1) / dl.Get(k-1)
			d.Set(k-1, dl.Get(k-1))
			temp = d.Get(k + 1 - 1)
			d.Set(k+1-1, du.Get(k-1)-mult*temp)
			if k < ((*n) - 1) {
				dl.Set(k-1, du.Get(k+1-1))
				du.Set(k+1-1, -mult*dl.Get(k-1))
			}
			du.Set(k-1, temp)
			for j = 1; j <= (*nrhs); j++ {
				temp = b.Get(k-1, j-1)
				b.Set(k-1, j-1, b.Get(k+1-1, j-1))
				b.Set(k+1-1, j-1, temp-mult*b.Get(k+1-1, j-1))
			}
		}
	}
	if d.Get((*n)-1) == zero {
		(*info) = (*n)
		return
	}

	//     Back solve with the matrix U from the factorization.
	for j = 1; j <= (*nrhs); j++ {
		b.Set((*n)-1, j-1, b.Get((*n)-1, j-1)/d.Get((*n)-1))
		if (*n) > 1 {
			b.Set((*n)-1-1, j-1, (b.Get((*n)-1-1, j-1)-du.Get((*n)-1-1)*b.Get((*n)-1, j-1))/d.Get((*n)-1-1))
		}
		for k = (*n) - 2; k >= 1; k-- {
			b.Set(k-1, j-1, (b.Get(k-1, j-1)-du.Get(k-1)*b.Get(k+1-1, j-1)-dl.Get(k-1)*b.Get(k+2-1, j-1))/d.Get(k-1))
		}
	}
}
