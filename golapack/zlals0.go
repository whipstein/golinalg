package golapack

import (
	"fmt"

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
func Zlals0(icompq, nl, nr, sqre, nrhs int, b, bx *mat.CMatrix, perm *[]int, givptr int, givcol *[]int, ldgcol int, givnum, poles *mat.Matrix, difl *mat.Vector, difr *mat.Matrix, z *mat.Vector, k int, c, s float64, rwork *mat.Vector) (err error) {
	var diflj, difrj, dj, dsigj, dsigjp, negone, one, temp, zero float64
	var i, j, jcol, jrow, m, n, nlp1 int

	one = 1.0
	zero = 0.0
	negone = -1.0

	//     Test the input parameters.
	n = nl + nr + 1

	if (icompq < 0) || (icompq > 1) {
		err = fmt.Errorf("(icompq < 0) || (icompq > 1): icompq=%v", icompq)
	} else if nl < 1 {
		err = fmt.Errorf("nl < 1: nl=%v", nl)
	} else if nr < 1 {
		err = fmt.Errorf("nr < 1: nr=%v", nr)
	} else if (sqre < 0) || (sqre > 1) {
		err = fmt.Errorf("(sqre < 0) || (sqre > 1): sqre=%v", sqre)
	} else if nrhs < 1 {
		err = fmt.Errorf("nrhs < 1: nrhs=%v", nrhs)
	} else if b.Rows < n {
		err = fmt.Errorf("b.Rows < n: b.Rows=%v, n=%v", b.Rows, n)
	} else if bx.Rows < n {
		err = fmt.Errorf("bx.Rows < n: bx.Rows=%v, n=%v", bx.Rows, n)
	} else if givptr < 0 {
		err = fmt.Errorf("givptr < 0: givptr=%v", givptr)
	} else if ldgcol < n {
		err = fmt.Errorf("ldgcol < n: ldgcol=%v", ldgcol)
	} else if givnum.Rows < n {
		err = fmt.Errorf("givnum.Rows < n: givnum.Rows=%v, n=%v", givnum.Rows, n)
	} else if k < 1 {
		err = fmt.Errorf("k < 1: k=%v", k)
	}
	if err != nil {
		gltest.Xerbla2("Zlals0", err)
		return
	}

	m = n + sqre
	nlp1 = nl + 1

	if icompq == 0 {
		//        Apply back orthogonal transformations from the left.
		//
		//        Step (1L): apply back the Givens rotations performed.
		for i = 1; i <= givptr; i++ {
			b.Off((*givcol)[i-1+(1-1)*ldgcol]-1, 0).CVector().Drot(nrhs, b.Off((*givcol)[i-1+(2-1)*ldgcol]-1, 0).CVector(), b.Rows, b.Rows, givnum.Get(i-1, 1), givnum.Get(i-1, 0))
		}

		//        Step (2L): permute rows of B.
		bx.Off(0, 0).CVector().Copy(nrhs, b.Off(nlp1-1, 0).CVector(), b.Rows, bx.Rows)
		for i = 2; i <= n; i++ {
			bx.Off(i-1, 0).CVector().Copy(nrhs, b.Off((*perm)[i-1]-1, 0).CVector(), b.Rows, bx.Rows)
		}

		//        Step (3L): apply the inverse of the left singular vector
		//        matrix to BX.
		if k == 1 {
			b.Off(0, 0).CVector().Copy(nrhs, bx.Off(0, 0).CVector(), bx.Rows, b.Rows)
			if z.Get(0) < zero {
				b.Off(0, 0).CVector().Dscal(nrhs, negone, b.Rows)
			}
		} else {
			for j = 1; j <= k; j++ {
				diflj = difl.Get(j - 1)
				dj = poles.Get(j-1, 0)
				dsigj = -poles.Get(j-1, 1)
				if j < k {
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
				for i = j + 1; i <= k; i++ {
					if (z.Get(i-1) == zero) || (poles.Get(i-1, 1) == zero) {
						rwork.Set(i-1, zero)
					} else {
						rwork.Set(i-1, poles.Get(i-1, 1)*z.Get(i-1)/(Dlamc3(poles.GetPtr(i-1, 1), &dsigjp)+difrj)/(poles.Get(i-1, 1)+dj))
					}
				}
				rwork.Set(0, negone)
				temp = rwork.Nrm2(k, 1)

				//              Since B and BX are complex, the following call to DGEMV
				//              is performed in two steps (real and imaginary parts).
				//
				//              CALL DGEMV( 'T', K, NRHS, ONE, BX, LDBX, WORK, 1, ZERO,
				//    $                     B( J, 1 ), LDB )
				i = k + nrhs*2
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = 1; jrow <= k; jrow++ {
						i = i + 1
						rwork.Set(i-1, bx.GetRe(jrow-1, jcol-1))
					}
				}
				if err = rwork.Off(1+k-1).Gemv(Trans, k, nrhs, one, rwork.Off(1+k+nrhs*2-1).Matrix(k, opts), rwork, 1, zero, 1); err != nil {
					panic(err)
				}
				i = k + nrhs*2
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = 1; jrow <= k; jrow++ {
						i = i + 1
						rwork.Set(i-1, bx.GetIm(jrow-1, jcol-1))
					}
				}
				if err = rwork.Off(1+k+nrhs-1).Gemv(Trans, k, nrhs, one, rwork.Off(1+k+nrhs*2-1).Matrix(k, opts), rwork, 1, zero, 1); err != nil {
					panic(err)
				}
				for jcol = 1; jcol <= nrhs; jcol++ {
					b.Set(j-1, jcol-1, complex(rwork.Get(jcol+k-1), rwork.Get(jcol+k+nrhs-1)))
				}
				if err = Zlascl('G', 0, 0, temp, one, 1, nrhs, b.Off(j-1, 0)); err != nil {
					panic(err)
				}
			}
		}

		//        Move the deflated rows of BX to B also.
		if k < max(m, n) {
			Zlacpy(Full, n-k, nrhs, bx.Off(k, 0), b.Off(k, 0))
		}
	} else {
		//        Apply back the right orthogonal transformations.
		//
		//        Step (1R): apply back the new right singular vector matrix
		//        to B.
		if k == 1 {
			bx.Off(0, 0).CVector().Copy(nrhs, b.Off(0, 0).CVector(), b.Rows, bx.Rows)
		} else {
			for j = 1; j <= k; j++ {
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
				for i = j + 1; i <= k; i++ {
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
				i = k + nrhs*2
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = 1; jrow <= k; jrow++ {
						i = i + 1
						rwork.Set(i-1, b.GetRe(jrow-1, jcol-1))
					}
				}
				if err = rwork.Off(1+k-1).Gemv(Trans, k, nrhs, one, rwork.Off(1+k+nrhs*2-1).Matrix(k, opts), rwork, 1, zero, 1); err != nil {
					panic(err)
				}
				i = k + nrhs*2
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = 1; jrow <= k; jrow++ {
						i = i + 1
						rwork.Set(i-1, b.GetIm(jrow-1, jcol-1))
					}
				}
				if err = rwork.Off(1+k+nrhs-1).Gemv(Trans, k, nrhs, one, rwork.Off(1+k+nrhs*2-1).Matrix(k, opts), rwork, 1, zero, 1); err != nil {
					panic(err)
				}
				for jcol = 1; jcol <= nrhs; jcol++ {
					bx.Set(j-1, jcol-1, complex(rwork.Get(jcol+k-1), rwork.Get(jcol+k+nrhs-1)))
				}
			}
		}

		//        Step (2R): if SQRE = 1, apply back the rotation that is
		//        related to the right null space of the subproblem.
		if sqre == 1 {
			bx.Off(m-1, 0).CVector().Copy(nrhs, b.Off(m-1, 0).CVector(), b.Rows, bx.Rows)
			bx.Off(m-1, 0).CVector().Drot(nrhs, bx.Off(0, 0).CVector(), bx.Rows, bx.Rows, c, s)
		}
		if k < max(m, n) {
			Zlacpy(Full, n-k, nrhs, b.Off(k, 0), bx.Off(k, 0))
		}

		//        Step (3R): permute rows of B.
		b.Off(nlp1-1, 0).CVector().Copy(nrhs, bx.Off(0, 0).CVector(), bx.Rows, b.Rows)
		if sqre == 1 {
			b.Off(m-1, 0).CVector().Copy(nrhs, bx.Off(m-1, 0).CVector(), bx.Rows, b.Rows)
		}
		for i = 2; i <= n; i++ {
			b.Off((*perm)[i-1]-1, 0).CVector().Copy(nrhs, bx.Off(i-1, 0).CVector(), bx.Rows, b.Rows)
		}

		//        Step (4R): apply back the Givens rotations performed.
		for i = givptr; i >= 1; i-- {
			b.Off((*givcol)[i-1+(1-1)*ldgcol]-1, 0).CVector().Drot(nrhs, b.Off((*givcol)[i-1+(2-1)*ldgcol]-1, 0).CVector(), b.Rows, b.Rows, givnum.Get(i-1, 1), -givnum.Get(i-1, 0))
		}
	}

	return
}
