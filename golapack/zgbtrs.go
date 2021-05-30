package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgbtrs solves a system of linear equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B
// with a general band matrix A using the LU factorization computed
// by ZGBTRF.
func Zgbtrs(trans byte, n, kl, ku, nrhs *int, ab *mat.CMatrix, ldab *int, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var lnoti, notran bool
	var one complex128
	var i, j, kd, l, lm int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	notran = trans == 'N'
	if !notran && trans != 'T' && trans != 'C' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kl) < 0 {
		(*info) = -3
	} else if (*ku) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -5
	} else if (*ldab) < (2*(*kl) + (*ku) + 1) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	kd = (*ku) + (*kl) + 1
	lnoti = (*kl) > 0

	if notran {
		//        Solve  A*X = B.
		//
		//        Solve L*X = B, overwriting B with X.
		//
		//        L is represented as a product of permutations and unit lower
		//        triangular matrices L = P(1) * L(1) * ... * P(n-1) * L(n-1),
		//        where each transformation L(i) is a rank-one modification of
		//        the identity matrix.
		if lnoti {
			for j = 1; j <= (*n)-1; j++ {
				lm = minint(*kl, (*n)-j)
				l = (*ipiv)[j-1]
				if l != j {
					goblas.Zswap(nrhs, b.CVector(l-1, 0), ldb, b.CVector(j-1, 0), ldb)
				}
				goblas.Zgeru(&lm, nrhs, toPtrc128(-one), ab.CVector(kd+1-1, j-1), func() *int { y := 1; return &y }(), b.CVector(j-1, 0), ldb, b.Off(j+1-1, 0), ldb)
			}
		}

		for i = 1; i <= (*nrhs); i++ {
			//           Solve U*X = B, overwriting B with X.
			goblas.Ztbsv(Upper, NoTrans, NonUnit, n, toPtr((*kl)+(*ku)), ab, ldab, b.CVector(0, i-1), func() *int { y := 1; return &y }())
		}

	} else if trans == 'T' {
		//        Solve A**T * X = B.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve U**T * X = B, overwriting B with X.
			goblas.Ztbsv(Upper, Trans, NonUnit, n, toPtr((*kl)+(*ku)), ab, ldab, b.CVector(0, i-1), func() *int { y := 1; return &y }())
		}

		//        Solve L**T * X = B, overwriting B with X.
		if lnoti {
			for j = (*n) - 1; j >= 1; j-- {
				lm = minint(*kl, (*n)-j)
				goblas.Zgemv(Trans, &lm, nrhs, toPtrc128(-one), b.Off(j+1-1, 0), ldb, ab.CVector(kd+1-1, j-1), func() *int { y := 1; return &y }(), &one, b.CVector(j-1, 0), ldb)
				l = (*ipiv)[j-1]
				if l != j {
					goblas.Zswap(nrhs, b.CVector(l-1, 0), ldb, b.CVector(j-1, 0), ldb)
				}
			}
		}

	} else {
		//        Solve A**H * X = B.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve U**H * X = B, overwriting B with X.
			goblas.Ztbsv(Upper, ConjTrans, NonUnit, n, toPtr((*kl)+(*ku)), ab, ldab, b.CVector(0, i-1), func() *int { y := 1; return &y }())
		}

		//        Solve L**H * X = B, overwriting B with X.
		if lnoti {
			for j = (*n) - 1; j >= 1; j-- {
				lm = minint(*kl, (*n)-j)
				Zlacgv(nrhs, b.CVector(j-1, 0), ldb)
				goblas.Zgemv(ConjTrans, &lm, nrhs, toPtrc128(-one), b.Off(j+1-1, 0), ldb, ab.CVector(kd+1-1, j-1), func() *int { y := 1; return &y }(), &one, b.CVector(j-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(j-1, 0), ldb)
				l = (*ipiv)[j-1]
				if l != j {
					goblas.Zswap(nrhs, b.CVector(l-1, 0), ldb, b.CVector(j-1, 0), ldb)
				}
			}
		}
	}
}
