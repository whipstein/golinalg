package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zgetc2 computes an LU factorization, using complete pivoting, of the
// n-by-n matrix A. The factorization has the form A = P * L * U * Q,
// where P and Q are permutation matrices, L is lower triangular with
// unit diagonal elements and U is upper triangular.
//
// This is a level 1 BLAS version of the algorithm.
func Zgetc2(n int, a *mat.CMatrix, ipiv, jpiv *[]int) (info int) {
	var bignum, eps, one, smin, smlnum, xmax, zero float64
	var i, ip, ipv, j, jp, jpv int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Handle the case N=1 by itself
	if n == 1 {
		(*ipiv)[0] = 1
		(*jpiv)[0] = 1
		if a.GetMag(0, 0) < smlnum {
			info = 1
			a.SetRe(0, 0, smlnum)
		}
		return
	}

	//     Factorize A using complete pivoting.
	//     Set pivots less than SMIN to SMIN
	for i = 1; i <= n-1; i++ {
		//        Find max element in matrix A
		xmax = zero
		for ip = i; ip <= n; ip++ {
			for jp = i; jp <= n; jp++ {
				if a.GetMag(ip-1, jp-1) >= xmax {
					xmax = a.GetMag(ip-1, jp-1)
					ipv = ip
					jpv = jp
				}
			}
		}
		if i == 1 {
			smin = math.Max(eps*xmax, smlnum)
		}

		//        Swap rows
		if ipv != i {
			a.Off(i-1, 0).CVector().Swap(n, a.Off(ipv-1, 0).CVector(), a.Rows, a.Rows)
		}
		(*ipiv)[i-1] = ipv

		//        Swap columns
		if jpv != i {
			a.Off(0, i-1).CVector().Swap(n, a.Off(0, jpv-1).CVector(), 1, 1)
		}
		(*jpiv)[i-1] = jpv

		//        Check for singularity
		if a.GetMag(i-1, i-1) < smin {
			info = i
			a.SetRe(i-1, i-1, smin)
		}
		for j = i + 1; j <= n; j++ {
			a.Set(j-1, i-1, a.Get(j-1, i-1)/a.Get(i-1, i-1))
		}
		if err = a.Off(i, i).Geru(n-i, n-i, -complex(one, 0), a.Off(i, i-1).CVector(), 1, a.Off(i-1, i).CVector(), a.Rows); err != nil {
			panic(err)
		}
	}

	if a.GetMag(n-1, n-1) < smin {
		info = n
		a.SetRe(n-1, n-1, smin)
	}

	//     Set last pivots to N
	(*ipiv)[n-1] = n
	(*jpiv)[n-1] = n

	return
}
