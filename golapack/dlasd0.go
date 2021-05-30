package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlasd0 Using a divide and conquer approach, DLASD0 computes the singular
// value decomposition (SVD) of a real upper bidiagonal N-by-M
// matrix B with diagonal D and offdiagonal E, where M = N + SQRE.
// The algorithm computes orthogonal matrices U and VT such that
// B = U * S * VT. The singular values S are overwritten on D.
//
// A related subroutine, DLASDA, computes only the singular values,
// and optionally, the singular vectors in compact form.
func Dlasd0(n, sqre *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt, smlsiz *int, iwork *[]int, work *mat.Vector, info *int) {
	var alpha, beta float64
	var i, i1, ic, idxq, idxqc, im1, inode, itemp, iwk, j, lf, ll, lvl, m, ncc, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqrei int

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -1
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -2
	}

	m = (*n) + (*sqre)

	if (*ldu) < (*n) {
		(*info) = -6
	} else if (*ldvt) < m {
		(*info) = -8
	} else if (*smlsiz) < 3 {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD0"), -(*info))
		return
	}

	//     If the input matrix is too small, call DLASDQ to find the SVD.
	if (*n) <= (*smlsiz) {
		Dlasdq('U', sqre, n, &m, n, toPtr(0), d, e, vt, ldvt, u, ldu, u, ldu, work, info)
		return
	}

	//     Set up the computation tree.
	inode = 1
	ndiml = inode + (*n)
	ndimr = ndiml + (*n)
	idxq = ndimr + (*n)
	iwk = idxq + (*n)
	Dlasdt(n, &nlvl, &nd, toSlice(iwork, inode-1), toSlice(iwork, ndiml-1), toSlice(iwork, ndimr-1), smlsiz)

	//     For the nodes on bottom level of the tree, solve
	//     their subproblems by DLASDQ.
	ndb1 = (nd + 1) / 2
	ncc = 0
	for i = ndb1; i <= nd; i++ {
		//     IC : center row of each node
		//     NL : number of rows of left  subproblem
		//     NR : number of rows of right subproblem
		//     NLF: starting row of the left   subproblem
		//     NRF: starting row of the right  subproblem
		i1 = i - 1
		ic = (*iwork)[inode+i1-1]
		nl = (*iwork)[ndiml+i1-1]
		nlp1 = nl + 1
		nr = (*iwork)[ndimr+i1-1]
		nrp1 = nr + 1
		nlf = ic - nl
		nrf = ic + 1
		sqrei = 1
		Dlasdq('U', &sqrei, &nl, &nlp1, &nl, &ncc, d.Off(nlf-1), e.Off(nlf-1), vt.Off(nlf-1, nlf-1), ldvt, u.Off(nlf-1, nlf-1), ldu, u.Off(nlf-1, nlf-1), ldu, work, info)
		if (*info) != 0 {
			return
		}
		itemp = idxq + nlf - 2
		for j = 1; j <= nl; j++ {
			(*iwork)[itemp+j-1] = j
		}
		if i == nd {
			sqrei = (*sqre)
		} else {
			sqrei = 1
		}
		nrp1 = nr + sqrei
		Dlasdq('U', &sqrei, &nr, &nrp1, &nr, &ncc, d.Off(nrf-1), e.Off(nrf-1), vt.Off(nrf-1, nrf-1), ldvt, u.Off(nrf-1, nrf-1), ldu, u.Off(nrf-1, nrf-1), ldu, work, info)
		if (*info) != 0 {
			return
		}
		itemp = idxq + ic
		for j = 1; j <= nr; j++ {
			(*iwork)[itemp+j-1-1] = j
		}
	}

	//     Now conquer each subproblem bottom-up.
	for lvl = nlvl; lvl >= 1; lvl-- {
		//        Find the first node LF and last node LL on the
		//        current level LVL.
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
			if ((*sqre) == 0) && (i == ll) {
				sqrei = (*sqre)
			} else {
				sqrei = 1
			}
			idxqc = idxq + nlf - 1
			alpha = d.Get(ic - 1)
			beta = e.Get(ic - 1)
			Dlasd1(&nl, &nr, &sqrei, d.Off(nlf-1), &alpha, &beta, u.Off(nlf-1, nlf-1), ldu, vt.Off(nlf-1, nlf-1), ldvt, toSlice(iwork, idxqc-1), toSlice(iwork, iwk-1), work, info)

			//        Report the possible convergence failure.
			if (*info) != 0 {
				return
			}
		}
	}
}
