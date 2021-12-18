package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlalsa is an itermediate step in solving the least squares problem
// by computing the SVD of the coefficient matrix in compact form (The
// singular vectors are computed as products of simple orthorgonal
// matrices.).
//
// If ICOMPQ = 0, Zlalsa applies the inverse of the left singular vector
// matrix of an upper bidiagonal matrix to the right hand side; and if
// ICOMPQ = 1, Zlalsa applies the right singular vector matrix to the
// right hand side. The singular vector matrices were generated in
// compact form by Zlalsa.
func Zlalsa(icompq, smlsiz, n, nrhs int, b, bx *mat.CMatrix, u, vt *mat.Matrix, k *[]int, difl, difr, z, poles *mat.Matrix, givptr, givcol *[]int, ldgcol int, perm *[]int, givnum *mat.Matrix, c, s, rwork *mat.Vector, iwork *[]int) (err error) {
	var one, zero float64
	var i, i1, ic, im1, inode, j, jcol, jimag, jreal, jrow, lf, ll, lvl, lvl2, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	if (icompq < 0) || (icompq > 1) {
		err = fmt.Errorf("(icompq < 0) || (icompq > 1): icompq=%v", icompq)
	} else if smlsiz < 3 {
		err = fmt.Errorf("smlsiz < 3: smlsiz=%v", smlsiz)
	} else if n < smlsiz {
		err = fmt.Errorf("n < smlsiz: n=%v, smlsiz=%v", n, smlsiz)
	} else if nrhs < 1 {
		err = fmt.Errorf("nrhs < 1: nrhs=%v", nrhs)
	} else if b.Rows < n {
		err = fmt.Errorf("b.Rows < n: b.Rows=%v, n=%v", b.Rows, n)
	} else if bx.Rows < n {
		err = fmt.Errorf("bx.Rows < n: bx.Rows=%v, n=%v", bx.Rows, n)
	} else if u.Rows < n {
		err = fmt.Errorf("u.Rows < n: u.Rows=%v, n=%v", u.Rows, n)
	} else if ldgcol < n {
		err = fmt.Errorf("ldgcol < n: ldgcol=%v, n=%v", ldgcol, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlalsa", err)
		return
	}

	//     Book-keeping and  setting up the computation tree.
	inode = 1
	ndiml = inode + n
	ndimr = ndiml + n

	nlvl, nd = Dlasdt(n, toSlice(iwork, inode-1), toSlice(iwork, ndiml-1), toSlice(iwork, ndimr-1), smlsiz)

	//     The following code applies back the left singular vector factors.
	//     For applying back the right singular vector factors, go to 170.
	if icompq == 1 {
		goto label170
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

		//        Since B and BX are complex, the following call to DGEMM
		//        is performed in two steps (real and imaginary parts).
		//
		//        CALL DGEMM( 'T', 'N', NL, NRHS, NL, ONE, U( NLF, 1 ), LDU,
		//     $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
		j = nl * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nl-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = rwork.Matrix(nl, opts).Gemm(Trans, NoTrans, nl, nrhs, nl, one, u.Off(nlf-1, 0), rwork.Off(1+nl*nrhs*2-1).Matrix(nl, opts), zero); err != nil {
			panic(err)
		}
		j = nl * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nl-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = rwork.Off(1+nl*nrhs-1).Matrix(nl, opts).Gemm(Trans, NoTrans, nl, nrhs, nl, one, u.Off(nlf-1, 0), rwork.Off(1+nl*nrhs*2-1).Matrix(nl, opts), zero); err != nil {
			panic(err)
		}
		jreal = 0
		jimag = nl * nrhs
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nl-1; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				bx.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		//        Since B and BX are complex, the following call to DGEMM
		//        is performed in two steps (real and imaginary parts).
		//
		//        CALL DGEMM( 'T', 'N', NR, NRHS, NR, ONE, U( NRF, 1 ), LDU,
		//    $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
		j = nr * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nr-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = rwork.Matrix(nr, opts).Gemm(Trans, NoTrans, nr, nrhs, nr, one, u.Off(nrf-1, 0), rwork.Off(1+nr*nrhs*2-1).Matrix(nr, opts), zero); err != nil {
			panic(err)
		}
		j = nr * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nr-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = rwork.Off(1+nr*nrhs-1).Matrix(nr, opts).Gemm(Trans, NoTrans, nr, nrhs, nr, one, u.Off(nrf-1, 0), rwork.Off(1+nr*nrhs*2-1).Matrix(nr, opts), zero); err != nil {
			panic(err)
		}
		jreal = 0
		jimag = nr * nrhs
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nr-1; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				bx.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

	}

	//     Next copy the rows of B that correspond to unchanged rows
	//     in the bidiagonal matrix to BX.
	for i = 1; i <= nd; i++ {
		ic = (*iwork)[inode+i-1-1]
		bx.Off(ic-1, 0).CVector().Copy(nrhs, b.Off(ic-1, 0).CVector(), b.Rows, bx.Rows)
	}

	//     Finally go through the left singular vector matrices of all
	//     the other subproblems bottom-up on the tree.
	j = pow(2, nlvl)
	sqre = 0

	for lvl = nlvl; lvl >= 1; lvl-- {
		lvl2 = 2*lvl - 1

		//        find the first node LF and last node LL on
		//        the current level LVL
		if lvl == 1 {
			lf = 1
			ll = 1
		} else {
			lf = pow(2, lvl-1)
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
			if err = Zlals0(icompq, nl, nr, sqre, nrhs, bx.Off(nlf-1, 0), b.Off(nlf-1, 0), toSlice(perm, nlf-1+(lvl-1)*ldgcol), (*givptr)[j-1], toSlice(givcol, nlf-1+(lvl2-1)*ldgcol), ldgcol, givnum.Off(nlf-1, lvl2-1), poles.Off(nlf-1, lvl2-1), difl.Off(nlf-1, lvl-1).Vector(), difr.Off(nlf-1, lvl2-1), z.Off(nlf-1, lvl-1).Vector(), (*k)[j-1], c.Get(j-1), s.Get(j-1), rwork); err != nil {
				panic(err)
			}
		}
	}
	return

	//     ICOMPQ = 1: applying back the right singular vector factors.
label170:
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
			lf = pow(2, lvl-1)
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
			if err = Zlals0(icompq, nl, nr, sqre, nrhs, b.Off(nlf-1, 0), bx.Off(nlf-1, 0), toSlice(perm, nlf-1+(lvl-1)*ldgcol), (*givptr)[j-1], toSlice(givcol, nlf-1+(lvl2-1)*ldgcol), ldgcol, givnum.Off(nlf-1, lvl2-1), poles.Off(nlf-1, lvl2-1), difl.Off(nlf-1, lvl-1).Vector(), difr.Off(nlf-1, lvl2-1), z.Off(nlf-1, lvl-1).Vector(), (*k)[j-1], c.Get(j-1), s.Get(j-1), rwork); err != nil {
				panic(err)
			}
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

		//        Since B and BX are complex, the following call to DGEMM is
		//        performed in two steps (real and imaginary parts).
		//
		//        CALL DGEMM( 'T', 'N', NLP1, NRHS, NLP1, ONE, VT( NLF, 1 ), LDU,
		//    $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
		j = nlp1 * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nlp1-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = rwork.Matrix(nlp1, opts).Gemm(Trans, NoTrans, nlp1, nrhs, nlp1, one, vt.Off(nlf-1, 0), rwork.Off(1+nlp1*nrhs*2-1).Matrix(nlp1, opts), zero); err != nil {
			panic(err)
		}
		j = nlp1 * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nlp1-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = rwork.Off(1+nlp1*nrhs-1).Matrix(nlp1, opts).Gemm(Trans, NoTrans, nlp1, nrhs, nlp1, one, vt.Off(nlf-1, 0), rwork.Off(1+nlp1*nrhs*2-1).Matrix(nlp1, opts), zero); err != nil {
			panic(err)
		}
		jreal = 0
		jimag = nlp1 * nrhs
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nlf; jrow <= nlf+nlp1-1; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				bx.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		//        Since B and BX are complex, the following call to DGEMM is
		//        performed in two steps (real and imaginary parts).
		//
		//        CALL DGEMM( 'T', 'N', NRP1, NRHS, NRP1, ONE, VT( NRF, 1 ), LDU,
		//    $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
		j = nrp1 * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nrp1-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = rwork.Matrix(nrp1, opts).Gemm(Trans, NoTrans, nrp1, nrhs, nrp1, one, vt.Off(nrf-1, 0), rwork.Off(1+nrp1*nrhs*2-1).Matrix(nrp1, opts), zero); err != nil {
			panic(err)
		}
		j = nrp1 * nrhs * 2
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nrp1-1; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = rwork.Off(1+nrp1*nrhs-1).Matrix(nrp1, opts).Gemm(Trans, NoTrans, nrp1, nrhs, nrp1, one, vt.Off(nrf-1, 0), rwork.Off(1+nrp1*nrhs*2-1).Matrix(nrp1, opts), zero); err != nil {
			panic(err)
		}
		jreal = 0
		jimag = nrp1 * nrhs
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = nrf; jrow <= nrf+nrp1-1; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				bx.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

	}

	return
}
