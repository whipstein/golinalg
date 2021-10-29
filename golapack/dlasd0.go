package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd0 Using a divide and conquer approach, Dlasd0 computes the singular
// value decomposition (SVD) of a real upper bidiagonal N-by-M
// matrix B with diagonal D and offdiagonal E, where M = N + SQRE.
// The algorithm computes orthogonal matrices U and VT such that
// B = U * S * VT. The singular values S are overwritten on D.
//
// A related subroutine, DLASDA, computes only the singular values,
// and optionally, the singular vectors in compact form.
func Dlasd0(n, sqre int, d, e *mat.Vector, u, vt *mat.Matrix, smlsiz int, iwork *[]int, work *mat.Vector) (info int, err error) {
	var alpha, beta float64
	var i, i1, ic, idxq, idxqc, im1, inode, itemp, iwk, j, lf, ll, lvl, m, ncc, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqrei int

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if (sqre < 0) || (sqre > 1) {
		err = fmt.Errorf("(sqre < 0) || (sqre > 1): sqre=%v", sqre)
	}

	m = n + sqre

	if u.Rows < n {
		err = fmt.Errorf("u.Rows < n: u.Rows=%v, n=%v", u.Rows, n)
	} else if vt.Rows < m {
		err = fmt.Errorf("vt.Rows < m: vt.Rows=%v, m=%v", vt.Rows, m)
	} else if smlsiz < 3 {
		err = fmt.Errorf("smlsiz < 3: smlsiz=%v", smlsiz)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd0", err)
		return
	}

	//     If the input matrix is too small, call DLASDQ to find the SVD.
	if n <= smlsiz {
		if info, err = Dlasdq(Upper, sqre, n, m, n, 0, d, e, vt, u, u, work); err != nil {
			panic(err)
		}
		return
	}

	//     Set up the computation tree.
	inode = 1
	ndiml = inode + n
	ndimr = ndiml + n
	idxq = ndimr + n
	iwk = idxq + n
	nlvl, nd = Dlasdt(n, toSlice(iwork, inode-1), toSlice(iwork, ndiml-1), toSlice(iwork, ndimr-1), smlsiz)

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
		if info, err = Dlasdq(Upper, sqrei, nl, nlp1, nl, ncc, d.Off(nlf-1), e.Off(nlf-1), vt.Off(nlf-1, nlf-1), u.Off(nlf-1, nlf-1), u.Off(nlf-1, nlf-1), work); err != nil {
			panic(err)
		}
		if info != 0 {
			return
		}
		itemp = idxq + nlf - 2
		for j = 1; j <= nl; j++ {
			(*iwork)[itemp+j-1] = j
		}
		if i == nd {
			sqrei = sqre
		} else {
			sqrei = 1
		}
		nrp1 = nr + sqrei
		if info, err = Dlasdq(Upper, sqrei, nr, nrp1, nr, ncc, d.Off(nrf-1), e.Off(nrf-1), vt.Off(nrf-1, nrf-1), u.Off(nrf-1, nrf-1), u.Off(nrf-1, nrf-1), work); err != nil {
			panic(err)
		}
		if info != 0 {
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
			if (sqre == 0) && (i == ll) {
				sqrei = sqre
			} else {
				sqrei = 1
			}
			idxqc = idxq + nlf - 1
			alpha = d.Get(ic - 1)
			beta = e.Get(ic - 1)
			if alpha, beta, info, err = Dlasd1(nl, nr, sqrei, d.Off(nlf-1), alpha, beta, u.Off(nlf-1, nlf-1), vt.Off(nlf-1, nlf-1), toSlice(iwork, idxqc-1), toSlice(iwork, iwk-1), work); err != nil {
				panic(err)
			}

			//        Report the possible convergence failure.
			if info != 0 {
				return
			}
		}
	}

	return
}
