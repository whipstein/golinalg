package golapack

import "golinalg/mat"

// Dlaswp performs a series of row interchanges on the matrix A.
// One row interchange is initiated for each of rows K1 through K2 of A.
func Dlaswp(n *int, a *mat.Matrix, lda *int, k1 *int, k2 *int, ipiv *[]int, incx *int) {
	var temp float64
	var i, i1, i2, inc, ip, ix, ix0, j, k, n32 int

	//     Interchange row I with row IPIV(K1+(I-K1)*abs(INCX)) for each of rows
	//     K1 through K2.
	if (*incx) > 0 {
		ix0 = (*k1)
		i1 = (*k1)
		i2 = (*k2)
		inc = 1
	} else if (*incx) < 0 {
		ix0 = (*k1) + ((*k1)-(*k2))*(*incx)
		i1 = (*k2)
		i2 = (*k1)
		inc = -1
	} else {
		return
	}

	n32 = ((*n) / 32) * 32
	if n32 != 0 {
		for j = 1; j <= n32; j += 32 {
			ix = ix0
			for _, i = range genIter(i1, i2, inc) {
				ip = (*ipiv)[ix-1]
				if ip != i {
					for k = j; k <= j+31; k++ {
						temp = a.Get(i-1, k-1)
						a.Set(i-1, k-1, a.Get(ip-1, k-1))
						a.Set(ip-1, k-1, temp)
					}
				}
				ix = ix + (*incx)
			}
		}
	}
	if n32 != (*n) {
		n32 = n32 + 1
		ix = ix0
		for _, i = range genIter(i1, i2, inc) {
			ip = (*ipiv)[ix-1]
			if ip != i {
				for k = n32; k <= (*n); k++ {
					temp = a.Get(i-1, k-1)
					a.Set(i-1, k-1, a.Get(ip-1, k-1))
					a.Set(ip-1, k-1, temp)
				}
			}
			ix = ix + (*incx)
		}
	}
}
