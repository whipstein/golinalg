package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlalsa is an itermediate step in solving the least squares problem
// by computing the SVD of the coefficient matrix in compact form (The
// singular vectors are computed as products of simple orthorgonal
// matrices.).
//
// If ICOMPQ = 0, DLALSA applies the inverse of the left singular vector
// matrix of an upper bidiagonal matrix to the right hand side; and if
// ICOMPQ = 1, DLALSA applies the right singular vector matrix to the
// right hand side. The singular vector matrices were generated in
// compact form by DLALSA.
func Dlalsa(icompq, smlsiz, n, nrhs *int, b *mat.Matrix, ldb *int, bx *mat.Matrix, ldbx *int, u *mat.Matrix, ldu *int, vt *mat.Matrix, k *[]int, difl, difr, z, poles *mat.Matrix, givptr *[]int, givcol *[]int, ldgcol *int, perm *[]int, givnum *mat.Matrix, c, s, work *mat.Vector, iwork *[]int, info *int) {
	var one, zero float64
	var i, i1, ic, im1, inode, j, lf, ll, lvl, lvl2, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0

	if ((*icompq) < 0) || ((*icompq) > 1) {
		(*info) = -1
	} else if (*smlsiz) < 3 {
		(*info) = -2
	} else if (*n) < (*smlsiz) {
		(*info) = -3
	} else if (*nrhs) < 1 {
		(*info) = -4
	} else if (*ldb) < (*n) {
		(*info) = -6
	} else if (*ldbx) < (*n) {
		(*info) = -8
	} else if (*ldu) < (*n) {
		(*info) = -10
	} else if (*ldgcol) < (*n) {
		(*info) = -19
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLALSA"), -(*info))
		return
	}

	//     Book-keeping and  setting up the computation tree.
	inode = 1
	ndiml = inode + (*n)
	ndimr = ndiml + (*n)

	Dlasdt(n, &nlvl, &nd, toSlice(iwork, inode-1), toSlice(iwork, ndiml-1), toSlice(iwork, ndimr-1), smlsiz)

	//     The following code applies back the left singular vector factors.
	//     For applying back the right singular vector factors, go to 50.
	if (*icompq) == 1 {
		goto label50
	}

	//     The nodes on the bottom level of the tree were solved
	//     by DLASDQ. The corresponding left and right singular vector
	//     matrices are in explicit form. First apply back the left
	//     singular vector matrices.
	ndb1 = (nd + 1) / 2
	for i = ndb1; i <= nd; i++ {
		//        IC : center row of each node
		//        NL : number of rows of left  subproblem
		//        NR : number of rows of right subproblem
		//        NLF: starting row of the left   subproblem
		//        NRF: starting row of the right  subproblem
		i1 = i - 1
		ic = (*iwork)[inode+i1-1]
		nl = (*iwork)[ndiml+i1-1]
		nr = (*iwork)[ndimr+i1-1]
		nlf = ic - nl
		nrf = ic + 1
		err = goblas.Dgemm(Trans, NoTrans, nl, *nrhs, nl, one, u.Off(nlf-1, 0), b.Off(nlf-1, 0), zero, bx.Off(nlf-1, 0))
		err = goblas.Dgemm(Trans, NoTrans, nr, *nrhs, nr, one, u.Off(nrf-1, 0), b.Off(nrf-1, 0), zero, bx.Off(nrf-1, 0))
	}

	//     Next copy the rows of B that correspond to unchanged rows
	//     in the bidiagonal matrix to BX.
	for i = 1; i <= nd; i++ {
		ic = (*iwork)[inode+i-1-1]
		goblas.Dcopy(*nrhs, b.Vector(ic-1, 0), bx.Vector(ic-1, 0))
	}

	//     Finally go through the left singular vector matrices of all
	//     the other subproblems bottom-up on the tree.
	j = int(math.Pow(2, float64(nlvl)))
	sqre = 0

	for lvl = nlvl; lvl >= 1; lvl-- {
		lvl2 = 2*lvl - 1

		//        find the first node LF and last node LL on
		//        the current level LVL
		if lvl == 1 {
			lf = 1
			ll = 1
		} else {
			lf = int(math.Pow(2, float64(lvl-1)))
			ll = 2*lf - 1
		}
		for i = lf; i <= ll; i++ {
			im1 = i - 1
			ic = (*iwork)[inode+im1-1]
			nl = (*iwork)[ndiml+im1-1]
			nr = (*iwork)[ndimr+im1-1]
			nlf = ic - nl
			nrf = ic + 1
			j = j - 1
			_permnlflvl := (*perm)[nlf-1+(lvl-1)*(*ldgcol):]
			_givcolnlflvl2 := (*givcol)[nlf-1+(lvl2-1)*(*ldgcol):]
			Dlals0(icompq, &nl, &nr, &sqre, nrhs, bx.Off(nlf-1, 0), ldbx, b.Off(nlf-1, 0), ldb, &_permnlflvl, &((*givptr)[j-1]), &_givcolnlflvl2, ldgcol, givnum.Off(nlf-1, lvl2-1), ldu, poles.Off(nlf-1, lvl2-1), difl.Vector(nlf-1, lvl-1), difr.Off(nlf-1, lvl2-1), z.Vector(nlf-1, lvl-1), &((*k)[j-1]), c.GetPtr(j-1), s.GetPtr(j-1), work, info)
		}
	}
	return

	//     ICOMPQ = 1: applying back the right singular vector factors.
label50:
	;

	//     First now go through the right singular vector matrices of all
	//     the tree nodes top-down.
	j = 0
	for lvl = 1; lvl <= nlvl; lvl++ {
		lvl2 = 2*lvl - 1

		//        Find the first node LF and last node LL on
		//        the current level LVL.
		if lvl == 1 {
			lf = 1
			ll = 1
		} else {
			lf = int(math.Pow(2, float64(lvl-1)))
			ll = 2*lf - 1
		}
		for i = ll; i >= lf; i-- {
			im1 = i - 1
			ic = (*iwork)[inode+im1-1]
			nl = (*iwork)[ndiml+im1-1]
			nr = (*iwork)[ndimr+im1-1]
			nlf = ic - nl
			nrf = ic + 1
			if i == ll {
				sqre = 0
			} else {
				sqre = 1
			}
			j = j + 1
			_permnlflvl := (*perm)[nlf-1+(lvl-1)*(*ldgcol):]
			_givcolnlflvl2 := (*givcol)[nlf-1+(lvl2-1)*(*ldgcol):]
			Dlals0(icompq, &nl, &nr, &sqre, nrhs, b.Off(nlf-1, 0), ldb, bx.Off(nlf-1, 0), ldbx, &_permnlflvl, &((*givptr)[j-1]), &_givcolnlflvl2, ldgcol, givnum.Off(nlf-1, lvl2-1), ldu, poles.Off(nlf-1, lvl2-1), difl.Vector(nlf-1, lvl-1), difr.Off(nlf-1, lvl2-1), z.Vector(nlf-1, lvl-1), &((*k)[j-1]), c.GetPtr(j-1), s.GetPtr(j-1), work, info)
		}
	}

	//     The nodes on the bottom level of the tree were solved
	//     by DLASDQ. The corresponding right singular vector
	//     matrices are in explicit form. Apply them back.
	ndb1 = (nd + 1) / 2
	for i = ndb1; i <= nd; i++ {
		i1 = i - 1
		ic = (*iwork)[inode+i1-1]
		nl = (*iwork)[ndiml+i1-1]
		nr = (*iwork)[ndimr+i1-1]
		nlp1 = nl + 1
		if i == nd {
			nrp1 = nr
		} else {
			nrp1 = nr + 1
		}
		nlf = ic - nl
		nrf = ic + 1
		err = goblas.Dgemm(Trans, NoTrans, nlp1, *nrhs, nlp1, one, vt.Off(nlf-1, 0), b.Off(nlf-1, 0), zero, bx.Off(nlf-1, 0))
		err = goblas.Dgemm(Trans, NoTrans, nrp1, *nrhs, nrp1, one, vt.Off(nrf-1, 0), b.Off(nrf-1, 0), zero, bx.Off(nrf-1, 0))
	}
}
