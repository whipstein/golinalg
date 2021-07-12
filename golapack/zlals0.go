package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlals0 applies back the multiplying factors of either the left or the
// right singular vector matrix of a diagonal matrix appended by a row
// to the right hand side matrix B in solving the least squares problem
// using the divide-and-conquer SVD approach.
//
// For the left singular vector matrix, three types of orthogonal
// matrices are involved:
//
// (1L) Givens rotations: the number of such rotations is GIVPTR; the
//      pairs of columns/rows they were applied to are stored in GIVCOL;
//      and the C- and S-values of these rotations are stored in GIVNUM.
//
// (2L) Permutation. The (NL+1)-st row of B is to be moved to the first
//      row, and for J=2:N, PERM(J)-th row of B is to be moved to the
//      J-th row.
//
// (3L) The left singular vector matrix of the remaining matrix.
//
// For the right singular vector matrix, four types of orthogonal
// matrices are involved:
//
// (1R) The right singular vector matrix of the remaining matrix.
//
// (2R) If SQRE = 1, one extra Givens rotation to generate the right
//      null space.
//
// (3R) The inverse transformation of (2L).
//
// (4R) The inverse transformation of (1L).
func Zlals0(icompq, nl, nr, sqre, nrhs *int, b *mat.CMatrix, ldb *int, bx *mat.CMatrix, ldbx *int, perm *[]int, givptr *int, givcol *[]int, ldgcol *int, givnum *mat.Matrix, ldgnum *int, poles *mat.Matrix, difl *mat.Vector, difr *mat.Matrix, z *mat.Vector, k *int, c, s *float64, rwork *mat.Vector, info *int) {
	var diflj, difrj, dj, dsigj, dsigjp, negone, one, temp, zero float64
	var i, j, jcol, jrow, m, n, nlp1 int
	var err error
	_ = err

	one = 1.0
	zero = 0.0
	negone = -1.0

	//     Test the input parameters.
	(*info) = 0
	n = (*nl) + (*nr) + 1

	if ((*icompq) < 0) || ((*icompq) > 1) {
		(*info) = -1
	} else if (*nl) < 1 {
		(*info) = -2
	} else if (*nr) < 1 {
		(*info) = -3
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -4
	} else if (*nrhs) < 1 {
		(*info) = -5
	} else if (*ldb) < n {
		(*info) = -7
	} else if (*ldbx) < n {
		(*info) = -9
	} else if (*givptr) < 0 {
		(*info) = -11
	} else if (*ldgcol) < n {
		(*info) = -13
	} else if (*ldgnum) < n {
		(*info) = -15
	} else if (*k) < 1 {
		(*info) = -20
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLALS0"), -(*info))
		return
	}

	m = n + (*sqre)
	nlp1 = (*nl) + 1

	if (*icompq) == 0 {
		//        Apply back orthogonal transformations from the left.
		//
		//        Step (1L): apply back the Givens rotations performed.
		for i = 1; i <= (*givptr); i++ {
			goblas.Zdrot(*nrhs, b.CVector((*givcol)[i-1+(2-1)*(*ldgcol)]-1, 0, *ldb), b.CVector((*givcol)[i-1+(1-1)*(*ldgcol)]-1, 0, *ldb), givnum.Get(i-1, 1), givnum.Get(i-1, 0))
		}

		//        Step (2L): permute rows of B.
		goblas.Zcopy(*nrhs, b.CVector(nlp1-1, 0, *ldb), bx.CVector(0, 0, *ldbx))
		for i = 2; i <= n; i++ {
			goblas.Zcopy(*nrhs, b.CVector((*perm)[i-1]-1, 0, *ldb), bx.CVector(i-1, 0, *ldbx))
		}

		//        Step (3L): apply the inverse of the left singular vector
		//        matrix to BX.
		if (*k) == 1 {
			goblas.Zcopy(*nrhs, bx.CVector(0, 0, *ldbx), b.CVector(0, 0, *ldb))
			if z.Get(0) < zero {
				goblas.Zdscal(*nrhs, negone, b.CVector(0, 0, *ldb))
			}
		} else {
			for j = 1; j <= (*k); j++ {
				diflj = difl.Get(j - 1)
				dj = poles.Get(j-1, 0)
				dsigj = -poles.Get(j-1, 1)
				if j < (*k) {
					difrj = -difr.Get(j-1, 0)
					dsigjp = -poles.Get(j, 1)
				}
				if (z.Get(j-1) == zero) || (poles.Get(j-1, 1) == zero) {
					rwork.Set(j-1, zero)
				} else {
					rwork.Set(j-1, -poles.Get(j-1, 1)*z.Get(j-1)/diflj/(poles.Get(j-1, 1)+dj))
				}
				for i = 1; i <= j-1; i++ {
					if (z.Get(i-1) == zero) || (poles.Get(i-1, 1) == zero) {
						rwork.Set(i-1, zero)
					} else {
						rwork.Set(i-1, poles.Get(i-1, 1)*z.Get(i-1)/(Dlamc3(poles.GetPtr(i-1, 1), &dsigj)-diflj)/(poles.Get(i-1, 1)+dj))
					}
				}
				for i = j + 1; i <= (*k); i++ {
					if (z.Get(i-1) == zero) || (poles.Get(i-1, 1) == zero) {
						rwork.Set(i-1, zero)
					} else {
						rwork.Set(i-1, poles.Get(i-1, 1)*z.Get(i-1)/(Dlamc3(poles.GetPtr(i-1, 1), &dsigjp)+difrj)/(poles.Get(i-1, 1)+dj))
					}
				}
				rwork.Set(0, negone)
				temp = goblas.Dnrm2(*k, rwork.Off(0, 1))

				//              Since B and BX are complex, the following call to DGEMV
				//              is performed in two steps (real and imaginary parts).
				//
				//              CALL DGEMV( 'T', K, NRHS, ONE, BX, LDBX, WORK, 1, ZERO,
				//    $                     B( J, 1 ), LDB )
				i = (*k) + (*nrhs)*2
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = 1; jrow <= (*k); jrow++ {
						i = i + 1
						rwork.Set(i-1, bx.GetRe(jrow-1, jcol-1))
					}
				}
				err = goblas.Dgemv(Trans, *k, *nrhs, one, rwork.MatrixOff(1+(*k)+(*nrhs)*2-1, *k, opts), rwork.Off(0, 1), zero, rwork.Off(1+(*k)-1, 1))
				i = (*k) + (*nrhs)*2
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = 1; jrow <= (*k); jrow++ {
						i = i + 1
						rwork.Set(i-1, bx.GetIm(jrow-1, jcol-1))
					}
				}
				err = goblas.Dgemv(Trans, *k, *nrhs, one, rwork.MatrixOff(1+(*k)+(*nrhs)*2-1, *k, opts), rwork.Off(0, 1), zero, rwork.Off(1+(*k)+(*nrhs)-1, 1))
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					b.Set(j-1, jcol-1, complex(rwork.Get(jcol+(*k)-1), rwork.Get(jcol+(*k)+(*nrhs)-1)))
				}
				Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &temp, &one, func() *int { y := 1; return &y }(), nrhs, b.Off(j-1, 0), ldb, info)
			}
		}

		//        Move the deflated rows of BX to B also.
		if (*k) < max(m, n) {
			Zlacpy('A', toPtr(n-(*k)), nrhs, bx.Off((*k), 0), ldbx, b.Off((*k), 0), ldb)
		}
	} else {
		//        Apply back the right orthogonal transformations.
		//
		//        Step (1R): apply back the new right singular vector matrix
		//        to B.
		if (*k) == 1 {
			goblas.Zcopy(*nrhs, b.CVector(0, 0, *ldb), bx.CVector(0, 0, *ldbx))
		} else {
			for j = 1; j <= (*k); j++ {
				dsigj = poles.Get(j-1, 1)
				if z.Get(j-1) == zero {
					rwork.Set(j-1, zero)
				} else {
					rwork.Set(j-1, -z.Get(j-1)/difl.Get(j-1)/(dsigj+poles.Get(j-1, 0))/difr.Get(j-1, 1))
				}
				for i = 1; i <= j-1; i++ {
					if z.Get(j-1) == zero {
						rwork.Set(i-1, zero)
					} else {
						rwork.Set(i-1, z.Get(j-1)/(Dlamc3(&dsigj, toPtrf64(-poles.Get(i, 1)))-difr.Get(i-1, 0))/(dsigj+poles.Get(i-1, 0))/difr.Get(i-1, 1))
					}
				}
				for i = j + 1; i <= (*k); i++ {
					if z.Get(j-1) == zero {
						rwork.Set(i-1, zero)
					} else {
						rwork.Set(i-1, z.Get(j-1)/(Dlamc3(&dsigj, toPtrf64(-poles.Get(i-1, 1)))-difl.Get(i-1))/(dsigj+poles.Get(i-1, 0))/difr.Get(i-1, 1))
					}
				}

				//              Since B and BX are complex, the following call to DGEMV
				//              is performed in two steps (real and imaginary parts).
				//
				//              CALL DGEMV( 'T', K, NRHS, ONE, B, LDB, WORK, 1, ZERO,
				//    $                     BX( J, 1 ), LDBX )
				i = (*k) + (*nrhs)*2
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = 1; jrow <= (*k); jrow++ {
						i = i + 1
						rwork.Set(i-1, b.GetRe(jrow-1, jcol-1))
					}
				}
				err = goblas.Dgemv(Trans, *k, *nrhs, one, rwork.MatrixOff(1+(*k)+(*nrhs)*2-1, *k, opts), rwork.Off(0, 1), zero, rwork.Off(1+(*k)-1, 1))
				i = (*k) + (*nrhs)*2
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = 1; jrow <= (*k); jrow++ {
						i = i + 1
						rwork.Set(i-1, b.GetIm(jrow-1, jcol-1))
					}
				}
				err = goblas.Dgemv(Trans, *k, *nrhs, one, rwork.MatrixOff(1+(*k)+(*nrhs)*2-1, *k, opts), rwork.Off(0, 1), zero, rwork.Off(1+(*k)+(*nrhs)-1, 1))
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					bx.Set(j-1, jcol-1, complex(rwork.Get(jcol+(*k)-1), rwork.Get(jcol+(*k)+(*nrhs)-1)))
				}
			}
		}

		//        Step (2R): if SQRE = 1, apply back the rotation that is
		//        related to the right null space of the subproblem.
		if (*sqre) == 1 {
			goblas.Zcopy(*nrhs, b.CVector(m-1, 0, *ldb), bx.CVector(m-1, 0, *ldbx))
			goblas.Zdrot(*nrhs, bx.CVector(0, 0, *ldbx), bx.CVector(m-1, 0, *ldbx), *c, *s)
		}
		if (*k) < max(m, n) {
			Zlacpy('A', toPtr(n-(*k)), nrhs, b.Off((*k), 0), ldb, bx.Off((*k), 0), ldbx)
		}

		//        Step (3R): permute rows of B.
		goblas.Zcopy(*nrhs, bx.CVector(0, 0, *ldbx), b.CVector(nlp1-1, 0, *ldb))
		if (*sqre) == 1 {
			goblas.Zcopy(*nrhs, bx.CVector(m-1, 0, *ldbx), b.CVector(m-1, 0, *ldb))
		}
		for i = 2; i <= n; i++ {
			goblas.Zcopy(*nrhs, bx.CVector(i-1, 0, *ldbx), b.CVector((*perm)[i-1]-1, 0, *ldb))
		}

		//        Step (4R): apply back the Givens rotations performed.
		for i = (*givptr); i >= 1; i-- {
			goblas.Zdrot(*nrhs, b.CVector((*givcol)[i-1+(2-1)*(*ldgcol)]-1, 0, *ldb), b.CVector((*givcol)[i-1+(1-1)*(*ldgcol)]-1, 0, *ldb), givnum.Get(i-1, 1), -givnum.Get(i-1, 0))
		}
	}
}
