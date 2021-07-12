package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgbtrs solves a system of linear equations
//    A * X = B  or  A**T * X = B
// with a general band matrix A using the LU factorization computed
// by DGBTRF.
func Dgbtrs(trans byte, n, kl, ku, nrhs *int, ab *mat.Matrix, ldab *int, ipiv *[]int, b *mat.Matrix, ldb *int, info *int) {
	var lnoti, notran bool
	var one float64
	var i, j, kd, l, lm int
	var err error
	_ = err

	one = 1.0

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
	} else if (*ldb) < max(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGBTRS"), -(*info))
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
				lm = min(*kl, (*n)-j)
				l = (*ipiv)[j-1]
				if l != j {
					goblas.Dswap(*nrhs, b.Vector(l-1, 0), b.Vector(j-1, 0))
				}
				err = goblas.Dger(lm, *nrhs, -one, ab.Vector(kd, j-1, 1), b.Vector(j-1, 0), b.Off(j, 0))
			}
		}

		for i = 1; i <= (*nrhs); i++ {
			//           Solve U*X = B, overwriting B with X.
			err = goblas.Dtbsv(mat.Upper, mat.NoTrans, mat.NonUnit, *n, (*kl)+(*ku), ab, b.Vector(0, i-1, 1))
		}

	} else {
		//        Solve A**T*X = B.
		for i = 1; i <= (*nrhs); i++ {
			//           Solve U**T*X = B, overwriting B with X.
			err = goblas.Dtbsv(mat.Upper, mat.Trans, mat.NonUnit, *n, (*kl)+(*ku), ab, b.Vector(0, i-1, 1))
		}

		//        Solve L**T*X = B, overwriting B with X.
		if lnoti {
			for j = (*n) - 1; j >= 1; j-- {
				lm = min(*kl, (*n)-j)
				err = goblas.Dgemv(mat.Trans, lm, *nrhs, -one, b.Off(j, 0), ab.Vector(kd, j-1, 1), one, b.Vector(j-1, 0))
				l = (*ipiv)[j-1]
				if l != j {
					goblas.Dswap(*nrhs, b.Vector(l-1, 0), b.Vector(j-1, 0))
				}
			}
		}
	}
}
