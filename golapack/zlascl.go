package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlascl multiplies the M by N complex matrix A by the real scalar
// CTO/CFROM.  This is done without over/underflow as long as the final
// result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
// A may be full, upper triangular, lower triangular, upper Hessenberg,
// or banded.
func Zlascl(_type byte, kl, ku *int, cfrom, cto *float64, m, n *int, a *mat.CMatrix, lda, info *int) {
	var done bool
	var bignum, cfrom1, cfromc, cto1, ctoc, mul, one, smlnum, zero float64
	var i, itype, j, k1, k2, k3, k4 int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0

	if _type == 'G' {
		itype = 0
	} else if _type == 'L' {
		itype = 1
	} else if _type == 'U' {
		itype = 2
	} else if _type == 'H' {
		itype = 3
	} else if _type == 'B' {
		itype = 4
	} else if _type == 'Q' {
		itype = 5
	} else if _type == 'Z' {
		itype = 6
	} else {
		itype = -1
	}
	//
	if itype == -1 {
		(*info) = -1
	} else if (*cfrom) == zero || Disnan(int(*cfrom)) {
		(*info) = -4
	} else if Disnan(int(*cto)) {
		(*info) = -5
	} else if (*m) < 0 {
		(*info) = -6
	} else if (*n) < 0 || (itype == 4 && (*n) != (*m)) || (itype == 5 && (*n) != (*m)) {
		(*info) = -7
	} else if itype <= 3 && (*lda) < maxint(1, *m) {
		(*info) = -9
	} else if itype >= 4 {
		if (*kl) < 0 || (*kl) > maxint((*m)-1, 0) {
			(*info) = -2
		} else if (*ku) < 0 || (*ku) > maxint((*n)-1, 0) || ((itype == 4 || itype == 5) && (*kl) != (*ku)) {
			(*info) = -3
		} else if (itype == 4 && (*lda) < (*kl)+1) || (itype == 5 && (*lda) < (*ku)+1) || (itype == 6 && (*lda) < 2*(*kl)+(*ku)+1) {
			(*info) = -9
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLASCL"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum

	cfromc = (*cfrom)
	ctoc = (*cto)

label10:
	;
	cfrom1 = cfromc * smlnum
	if cfrom1 == cfromc {
		//!        CFROMC is an inf.  Multiply by a correctly signed zero for

		//!        finite CTOC, or a NaN if CTOC is infinite.

		mul = ctoc / cfromc
		done = true
		cto1 = ctoc
	} else {
		cto1 = ctoc / bignum
		if cto1 == ctoc {
			//!           CTOC is either 0 or an inf.  In both cases, CTOC itself

			//!           serves as the correct multiplication factor.

			mul = ctoc
			done = true
			cfromc = one
		} else if math.Abs(cfrom1) > math.Abs(ctoc) && ctoc != zero {
			mul = smlnum
			done = false
			cfromc = cfrom1
		} else if math.Abs(cto1) > math.Abs(cfromc) {
			mul = bignum
			done = false
			ctoc = cto1
		} else {
			mul = ctoc / cfromc
			done = true
		}
	}

	if itype == 0 {
		//        Full matrix
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 1 {
		//        Lower triangular matrix
		for j = 1; j <= (*n); j++ {
			for i = j; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 2 {
		//        Upper triangular matrix
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(j, *m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 3 {
		//        Upper Hessenberg matrix
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(j+1, *m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 4 {
		//        Lower half of a symmetric band matrix
		k3 = (*kl) + 1
		k4 = (*n) + 1
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= minint(k3, k4-j); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 5 {
		//        Upper half of a symmetric band matrix
		k1 = (*ku) + 2
		k3 = (*ku) + 1
		for j = 1; j <= (*n); j++ {
			for i = maxint(k1-j, 1); i <= k3; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	} else if itype == 6 {
		//        Band matrix
		k1 = (*kl) + (*ku) + 2
		k2 = (*kl) + 1
		k3 = 2*(*kl) + (*ku) + 1
		k4 = (*kl) + (*ku) + 1 + (*m)
		for j = 1; j <= (*n); j++ {
			for i = maxint(k1-j, k2); i <= minint(k3, k4-j); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(mul, 0))
			}
		}

	}

	if !done {
		goto label10
	}
}
