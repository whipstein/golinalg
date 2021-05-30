package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlasda Using a divide and conquer approach, DLASDA computes the singular
// value decomposition (SVD) of a real upper bidiagonal N-by-M matrix
// B with diagonal D and offdiagonal E, where M = N + SQRE. The
// algorithm computes the singular values in the SVD B = U * S * VT.
// The orthogonal matrices U and VT are optionally computed in
// compact form.
//
// A related subroutine, DLASD0, computes the singular values and
// the singular vectors in explicit form.
func Dlasda(icompq, smlsiz, n, sqre *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, k *[]int, difl, difr, z, poles *mat.Matrix, givptr, givcol *[]int, ldgcol *int, perm *[]int, givnum *mat.Matrix, c, s, work *mat.Vector, iwork *[]int, info *int) {
	var alpha, beta, one, zero float64
	var i, i1, ic, idxq, idxqi, im1, inode, itemp, iwk, j, lf, ll, lvl, lvl2, m, ncc, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, nru, nwork1, nwork2, smlszp, sqrei, vf, vfi, vl, vli int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0

	if ((*icompq) < 0) || ((*icompq) > 1) {
		(*info) = -1
	} else if (*smlsiz) < 3 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -4
	} else if (*ldu) < ((*n) + (*sqre)) {
		(*info) = -8
	} else if (*ldgcol) < (*n) {
		(*info) = -17
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASDA"), -(*info))
		return
	}

	m = (*n) + (*sqre)

	//     If the input matrix is too small, call DLASDQ to find the SVD.
	if (*n) <= (*smlsiz) {
		if (*icompq) == 0 {
			Dlasdq('U', sqre, n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, vt, ldu, u, ldu, u, ldu, work, info)
		} else {
			Dlasdq('U', sqre, n, &m, n, func() *int { y := 0; return &y }(), d, e, vt, ldu, u, ldu, u, ldu, work, info)
		}
		return
	}

	//     Book-keeping and  set up the computation tree.
	inode = 1
	ndiml = inode + (*n)
	ndimr = ndiml + (*n)
	idxq = ndimr + (*n)
	iwk = idxq + (*n)

	ncc = 0
	nru = 0

	smlszp = (*smlsiz) + 1
	vf = 1
	vl = vf + m
	nwork1 = vl + m
	nwork2 = nwork1 + smlszp*smlszp

	Dlasdt(n, &nlvl, &nd, toSlice(iwork, inode-1), toSlice(iwork, ndiml-1), toSlice(iwork, ndimr-1), smlsiz)

	//     for the nodes on bottom level of the tree, solve
	//     their subproblems by DLASDQ.
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
		nlp1 = nl + 1
		nr = (*iwork)[ndimr+i1-1]
		nlf = ic - nl
		nrf = ic + 1
		idxqi = idxq + nlf - 2
		vfi = vf + nlf - 1
		vli = vl + nlf - 1
		sqrei = 1
		if (*icompq) == 0 {
			Dlaset('A', &nlp1, &nlp1, &zero, &one, work.MatrixOff(nwork1-1, smlszp, opts), &smlszp)
			Dlasdq('U', &sqrei, &nl, &nlp1, &nru, &ncc, d.Off(nlf-1), e.Off(nlf-1), work.MatrixOff(nwork1-1, smlszp, opts), &smlszp, work.MatrixOff(nwork2-1, nl, opts), &nl, work.MatrixOff(nwork2-1, nl, opts), &nl, work.Off(nwork2-1), info)
			itemp = nwork1 + nl*smlszp
			goblas.Dcopy(&nlp1, work.Off(nwork1-1), toPtr(1), work.Off(vfi-1), toPtr(1))
			goblas.Dcopy(&nlp1, work.Off(itemp-1), toPtr(1), work.Off(vli-1), toPtr(1))
		} else {
			Dlaset('A', &nl, &nl, &zero, &one, u.Off(nlf-1, 0), ldu)
			Dlaset('A', &nlp1, &nlp1, &zero, &one, vt.Off(nlf-1, 0), ldu)
			Dlasdq('U', &sqrei, &nl, &nlp1, &nl, &ncc, d.Off(nlf-1), e.Off(nlf-1), vt.Off(nlf-1, 0), ldu, u.Off(nlf-1, 0), ldu, u.Off(nlf-1, 0), ldu, work.Off(nwork1-1), info)
			goblas.Dcopy(&nlp1, vt.Vector(nlf-1, 0), toPtr(1), work.Off(vfi-1), toPtr(1))
			goblas.Dcopy(&nlp1, vt.Vector(nlf-1, nlp1-1), toPtr(1), work.Off(vli-1), toPtr(1))
		}
		if (*info) != 0 {
			return
		}
		for j = 1; j <= nl; j++ {
			(*iwork)[idxqi+j-1] = j
		}
		if (i == nd) && ((*sqre) == 0) {
			sqrei = 0
		} else {
			sqrei = 1
		}
		idxqi = idxqi + nlp1
		vfi = vfi + nlp1
		vli = vli + nlp1
		nrp1 = nr + sqrei
		if (*icompq) == 0 {
			Dlaset('A', &nrp1, &nrp1, &zero, &one, work.MatrixOff(nwork1-1, smlszp, opts), &smlszp)
			Dlasdq('U', &sqrei, &nr, &nrp1, &nru, &ncc, d.Off(nrf-1), e.Off(nrf-1), work.MatrixOff(nwork1-1, smlszp, opts), &smlszp, work.MatrixOff(nwork2-1, nr, opts), &nr, work.MatrixOff(nwork2-1, nr, opts), &nr, work.Off(nwork2-1), info)
			itemp = nwork1 + (nrp1-1)*smlszp
			goblas.Dcopy(&nrp1, work.Off(nwork1-1), toPtr(1), work.Off(vfi-1), toPtr(1))
			goblas.Dcopy(&nrp1, work.Off(itemp-1), toPtr(1), work.Off(vli-1), toPtr(1))
		} else {
			Dlaset('A', &nr, &nr, &zero, &one, u.Off(nrf-1, 0), ldu)
			Dlaset('A', &nrp1, &nrp1, &zero, &one, vt.Off(nrf-1, 0), ldu)
			Dlasdq('U', &sqrei, &nr, &nrp1, &nr, &ncc, d.Off(nrf-1), e.Off(nrf-1), vt.Off(nrf-1, 0), ldu, u.Off(nrf-1, 0), ldu, u.Off(nrf-1, 0), ldu, work.Off(nwork1-1), info)
			goblas.Dcopy(&nrp1, vt.Vector(nrf-1, 0), toPtr(1), work.Off(vfi-1), toPtr(1))
			goblas.Dcopy(&nrp1, vt.Vector(nrf-1, nrp1-1), toPtr(1), work.Off(vli-1), toPtr(1))
		}
		if (*info) != 0 {
			return
		}
		for j = 1; j <= nr; j++ {
			(*iwork)[idxqi+j-1] = j
		}
	}

	//     Now conquer each subproblem bottom-up.
	j = int(math.Pow(2, float64(nlvl)))
	for lvl = nlvl; lvl >= 1; lvl-- {
		lvl2 = lvl*2 - 1

		//        Find the first node LF and last node LL on
		//        the current level LVL.
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
			if i == ll {
				sqrei = (*sqre)
			} else {
				sqrei = 1
			}
			vfi = vf + nlf - 1
			vli = vl + nlf - 1
			idxqi = idxq + nlf - 1
			alpha = d.Get(ic - 1)
			beta = e.Get(ic - 1)
			if (*icompq) == 0 {
				Dlasd6(icompq, &nl, &nr, &sqrei, d.Off(nlf-1), work.Off(vfi-1), work.Off(vli-1), &alpha, &beta, toSlice(iwork, idxqi-1), perm, &(*givptr)[0], givcol, ldgcol, givnum, ldu, poles, difl.VectorIdx(0), difr.VectorIdx(0), z.VectorIdx(0), &(*k)[0], c.GetPtr(0), s.GetPtr(0), work.Off(nwork1-1), toSlice(iwork, iwk-1), info)
			} else {
				j = j - 1
				Dlasd6(icompq, &nl, &nr, &sqrei, d.Off(nlf-1), work.Off(vfi-1), work.Off(vli-1), &alpha, &beta, toSlice(iwork, idxqi-1), toSlice(perm, nlf-1+(lvl-1)*(*ldgcol)), &(*givptr)[j-1], toSlice(givcol, nlf-1+(lvl2-1)*(*ldgcol)), ldgcol, givnum.Off(nlf-1, lvl2-1), ldu, poles.Off(nlf-1, lvl2-1), difl.Vector(nlf-1, lvl-1), difr.Vector(nlf-1, lvl2-1), z.Vector(nlf-1, lvl-1), &((*k)[j-1]), c.GetPtr(j-1), s.GetPtr(j-1), work.Off(nwork1-1), toSlice(iwork, iwk-1), info)
			}
			if (*info) != 0 {
				return
			}
		}
	}
}
