package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dgetc2 computes an LU factorization with complete pivoting of the
// n-by-n matrix A. The factorization has the form A = P * L * U * Q,
// where P and Q are permutation matrices, L is lower triangular with
// unit diagonal elements and U is upper triangular.
//
// This is the Level 2 BLAS algorithm.
func Dgetc2(n *int, a *mat.Matrix, lda *int, ipiv, jpiv *[]int, info *int) {
	var bignum, eps, one, smin, smlnum, xmax, zero float64
	var i, ip, ipv, j, jp, jpv int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	(*info) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Handle the case N=1 by itself
	if (*n) == 1 {
		(*ipiv)[0] = 1
		(*jpiv)[0] = 1
		if math.Abs(a.Get(0, 0)) < smlnum {
			(*info) = 1
			a.Set(0, 0, smlnum)
		}
		return
	}

	//     Factorize A using complete pivoting.
	//     Set pivots less than SMIN to SMIN.
	for i = 1; i <= (*n)-1; i++ {
		//        Find max element in matrix A
		xmax = zero
		for ip = i; ip <= (*n); ip++ {
			for jp = i; jp <= (*n); jp++ {
				if math.Abs(a.Get(ip-1, jp-1)) >= xmax {
					xmax = math.Abs(a.Get(ip-1, jp-1))
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
			goblas.Dswap(*n, a.Vector(ipv-1, 0), a.Vector(i-1, 0))
		}
		(*ipiv)[i-1] = ipv

		//        Swap columns
		if jpv != i {
			goblas.Dswap(*n, a.Vector(0, jpv-1, 1), a.Vector(0, i-1, 1))
		}
		(*jpiv)[i-1] = jpv

		//        Check for singularity
		if math.Abs(a.Get(i-1, i-1)) < smin {
			(*info) = i
			a.Set(i-1, i-1, smin)
		}
		for j = i + 1; j <= (*n); j++ {
			a.Set(j-1, i-1, a.Get(j-1, i-1)/a.Get(i-1, i-1))
		}
		err = goblas.Dger((*n)-i, (*n)-i, -one, a.Vector(i, i-1, 1), a.Vector(i-1, i, *lda), a.Off(i, i))
	}

	if math.Abs(a.Get((*n)-1, (*n)-1)) < smin {
		(*info) = (*n)
		a.Set((*n)-1, (*n)-1, smin)
	}

	//     Set last pivots to N
	(*ipiv)[(*n)-1] = (*n)
	(*jpiv)[(*n)-1] = (*n)
}
