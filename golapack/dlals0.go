package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlals0 applies back the multiplying factors of either the left or the
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
func Dlals0(icompq, nl, nr, sqre, nrhs *int, b *mat.Matrix, ldb *int, bx *mat.Matrix, ldbx *int, perm *[]int, givptr *int, givcol *[]int, ldgcol *int, givnum *mat.Matrix, ldgnum *int, poles *mat.Matrix, difl *mat.Vector, difr *mat.Matrix, z *mat.Vector, k *int, c, s *float64, work *mat.Vector, info *int) {
	var diflj, difrj, dj, dsigj, dsigjp, negone, one, temp, zero float64
	var i, j, m, n, nlp1 int

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
		gltest.Xerbla([]byte("DLALS0"), -(*info))
		return
	}

	m = n + (*sqre)
	nlp1 = (*nl) + 1

	if (*icompq) == 0 {
		//        Apply back orthogonal transformations from the left.
		//
		//        Step (1L): apply back the Givens rotations performed.
		for i = 1; i <= (*givptr); i++ {
			goblas.Drot(nrhs, b.Vector((*givcol)[i-1+(2-1)*(*ldgcol)]-1, 0), ldb, b.Vector((*givcol)[i-1+(1-1)*(*ldgcol)]-1, 0), ldb, givnum.GetPtr(i-1, 1), givnum.GetPtr(i-1, 0))
		}

		//        Step (2L): permute rows of B.
		goblas.Dcopy(nrhs, b.Vector(nlp1-1, 0), ldb, bx.Vector(0, 0), ldbx)
		for i = 2; i <= n; i++ {
			goblas.Dcopy(nrhs, b.Vector((*perm)[i-1]-1, 0), ldb, bx.Vector(i-1, 0), ldbx)
		}

		//        Step (3L): apply the inverse of the left singular vector
		//        matrix to BX.
		if (*k) == 1 {
			goblas.Dcopy(nrhs, bx.VectorIdx(0), ldbx, b.VectorIdx(0), ldb)
			if z.Get(0) < zero {
				goblas.Dscal(nrhs, &negone, b.VectorIdx(0), ldb)
			}
		} else {
			for j = 1; j <= (*k); j++ {
				diflj = difl.Get(j - 1)
				dj = poles.Get(j-1, 0)
				dsigj = -poles.Get(j-1, 1)
				if j < (*k) {
					difrj = -difr.Get(j-1, 0)
					dsigjp = -poles.Get(j+1-1, 1)
				}
				if (z.Get(j-1) == zero) || (poles.Get(j-1, 1) == zero) {
					work.Set(j-1, zero)
				} else {
					work.Set(j-1, -poles.Get(j-1, 1)*z.Get(j-1)/diflj/(poles.Get(j-1, 1)+dj))
				}
				for i = 1; i <= j-1; i++ {
					if (z.Get(i-1) == zero) || (poles.Get(i-1, 1) == zero) {
						work.Set(i-1, zero)
					} else {
						work.Set(i-1, poles.Get(i-1, 1)*z.Get(i-1)/(Dlamc3(poles.GetPtr(i-1, 1), &dsigj)-diflj)/(poles.Get(i-1, 1)+dj))
					}
				}
				for i = j + 1; i <= (*k); i++ {
					if (z.Get(i-1) == zero) || (poles.Get(i-1, 1) == zero) {
						work.Set(i-1, zero)
					} else {
						work.Set(i-1, poles.Get(i-1, 1)*z.Get(i-1)/(Dlamc3(poles.GetPtr(i-1, 1), &dsigjp)+difrj)/(poles.Get(i-1, 1)+dj))
					}
				}
				work.Set(0, negone)
				temp = goblas.Dnrm2(k, work, toPtr(1))
				goblas.Dgemv(Trans, k, nrhs, &one, bx, ldbx, work, toPtr(1), &zero, b.Vector(j-1, 0), ldb)
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &temp, &one, func() *int { y := 1; return &y }(), nrhs, b.Off(j-1, 0), ldb, info)
			}
		}

		//        Move the deflated rows of BX to B also.
		if (*k) < maxint(m, n) {
			Dlacpy('A', toPtr(n-(*k)), nrhs, bx.Off((*k)+1-1, 0), ldbx, b.Off((*k)+1-1, 0), ldb)
		}
	} else {
		//        Apply back the right orthogonal transformations.
		//
		//        Step (1R): apply back the new right singular vector matrix
		//        to B.
		if (*k) == 1 {
			goblas.Dcopy(nrhs, b.VectorIdx(0), ldb, bx.VectorIdx(0), ldbx)
		} else {
			for j = 1; j <= (*k); j++ {
				dsigj = poles.Get(j-1, 1)
				if z.Get(j-1) == zero {
					work.Set(j-1, zero)
				} else {
					work.Set(j-1, -z.Get(j-1)/difl.Get(j-1)/(dsigj+poles.Get(j-1, 0))/difr.Get(j-1, 1))
				}
				for i = 1; i <= j-1; i++ {
					if z.Get(j-1) == zero {
						work.Set(i-1, zero)
					} else {
						work.Set(i-1, z.Get(j-1)/(Dlamc3(&dsigj, func() *float64 { y := -poles.Get(i+1-1, 1); return &y }())-difr.Get(i-1, 0))/(dsigj+poles.Get(i-1, 0))/difr.Get(i-1, 1))
					}
				}
				for i = j + 1; i <= (*k); i++ {
					if z.Get(j-1) == zero {
						work.Set(i-1, zero)
					} else {
						work.Set(i-1, z.Get(j-1)/(Dlamc3(&dsigj, func() *float64 { y := -poles.Get(i-1, 1); return &y }())-difl.Get(i-1))/(dsigj+poles.Get(i-1, 0))/difr.Get(i-1, 1))
					}
				}
				goblas.Dgemv(Trans, k, nrhs, &one, b, ldb, work, toPtr(1), &zero, bx.Vector(j-1, 0), ldbx)
			}
		}

		//        Step (2R): if SQRE = 1, apply back the rotation that is
		//        related to the right null space of the subproblem.
		if (*sqre) == 1 {
			goblas.Dcopy(nrhs, b.Vector(m-1, 0), ldb, bx.Vector(m-1, 0), ldbx)
			goblas.Drot(nrhs, bx.Vector(0, 0), ldbx, bx.Vector(m-1, 0), ldbx, c, s)
		}
		if (*k) < maxint(m, n) {
			Dlacpy('A', toPtr(n-(*k)), nrhs, b.Off((*k)+1-1, 0), ldb, bx.Off((*k)+1-1, 0), ldbx)
		}

		//        Step (3R): permute rows of B.
		goblas.Dcopy(nrhs, bx.VectorIdx(0), ldbx, b.Vector(nlp1-1, 0), ldb)
		if (*sqre) == 1 {
			goblas.Dcopy(nrhs, bx.Vector(m-1, 0), ldbx, b.Vector(m-1, 0), ldb)
		}
		for i = 2; i <= n; i++ {
			goblas.Dcopy(nrhs, bx.Vector(i-1, 0), ldbx, b.Vector((*perm)[i-1]-1, 0), ldb)
		}

		//        Step (4R): apply back the Givens rotations performed.
		for i = (*givptr); i >= 1; i-- {
			goblas.Drot(nrhs, b.Vector((*givcol)[i-1+(2-1)*(*ldgcol)]-1, 0), ldb, b.Vector((*givcol)[i-1+(1-1)*(*ldgcol)]-1, 0), ldb, givnum.GetPtr(i-1, 1), toPtrf64(-givnum.Get(i-1, 0)))
		}
	}
}
