package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dsbgvx computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x.  Here A and B are assumed to be symmetric
// and banded, and B is also positive definite.  Eigenvalues and
// eigenvectors can be selected by specifying either all eigenvalues,
// a _range of values or a _range of indices for the desired eigenvalues.
func Dsbgvx(jobz, _range, uplo byte, n, ka, kb *int, ab *mat.Matrix, ldab *int, bb *mat.Matrix, ldbb *int, q *mat.Matrix, ldq *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, test, upper, valeig, wantz bool
	var order, vect byte
	var one, tmp1, zero float64
	var i, iinfo, indd, inde, indee, indibl, indisp, indiwo, indwrk, itmp1, j, jj, nsplit int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == 'U'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(alleig || valeig || indeig) {
		(*info) = -2
	} else if !(upper || uplo == 'L') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ka) < 0 {
		(*info) = -5
	} else if (*kb) < 0 || (*kb) > (*ka) {
		(*info) = -6
	} else if (*ldab) < (*ka)+1 {
		(*info) = -8
	} else if (*ldbb) < (*kb)+1 {
		(*info) = -10
	} else if (*ldq) < 1 || (wantz && (*ldq) < (*n)) {
		(*info) = -12
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -14
			}
		} else if indeig {
			if (*il) < 1 || (*il) > maxint(1, *n) {
				(*info) = -15
			} else if (*iu) < minint(*n, *il) || (*iu) > (*n) {
				(*info) = -16
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -21
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBGVX"), -(*info))
		return
	}

	//     Quick return if possible
	(*m) = 0
	if (*n) == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	Dpbstf(uplo, n, kb, bb, ldbb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem.
	Dsbgst(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, work, &iinfo)

	//     Reduce symmetric band matrix to tridiagonal form.
	indd = 1
	inde = indd + (*n)
	indwrk = inde + (*n)
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	Dsbtrd(vect, uplo, n, ka, ab, ldab, work.Off(indd-1), work.Off(inde-1), q, ldq, work.Off(indwrk-1), &iinfo)

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or SSTEQR.  If this fails for some
	//     eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ((*abstol) <= zero) {
		goblas.Dcopy(n, work.Off(indd-1), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }())
		indee = indwrk + 2*(*n)
		goblas.Dcopy(toPtr((*n)-1), work.Off(inde-1), func() *int { y := 1; return &y }(), work.Off(indee-1), func() *int { y := 1; return &y }())
		if !wantz {
			Dsterf(n, w, work.Off(indee-1), info)
		} else {
			Dlacpy('A', n, n, q, ldq, z, ldz)
			Dsteqr(jobz, n, w, work.Off(indee-1), z, ldz, work.Off(indwrk-1), info)
			if (*info) == 0 {
				for i = 1; i <= (*n); i++ {
					(*ifail)[i-1] = 0
				}
			}
		}
		if (*info) == 0 {
			(*m) = (*n)
			goto label30
		}
		(*info) = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired,
	//     call DSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + (*n)
	indiwo = indisp + (*n)
	Dstebz(_range, order, n, vl, vu, il, iu, abstol, work.Off(indd-1), work.Off(inde-1), m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwrk-1), toSlice(iwork, indiwo-1), info)

	if wantz {
		Dstein(n, work.Off(indd-1), work.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, work.Off(indwrk-1), toSlice(iwork, indiwo-1), ifail, info)

		//        Apply transformation matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by DSTEIN.
		for j = 1; j <= (*m); j++ {
			goblas.Dcopy(n, z.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
			goblas.Dgemv(NoTrans, n, n, &one, q, ldq, work, func() *int { y := 1; return &y }(), &zero, z.Vector(0, j-1), func() *int { y := 1; return &y }())
		}
	}

label30:
	;

	//     If eigenvalues are not in order, then sort them, along with
	//     eigenvectors.
	if wantz {
		for j = 1; j <= (*m)-1; j++ {
			i = 0
			tmp1 = w.Get(j - 1)
			for jj = j + 1; jj <= (*m); jj++ {
				if w.Get(jj-1) < tmp1 {
					i = jj
					tmp1 = w.Get(jj - 1)
				}
			}

			if i != 0 {
				itmp1 = (*iwork)[indibl+i-1-1]
				w.Set(i-1, w.Get(j-1))
				(*iwork)[indibl+i-1-1] = (*iwork)[indibl+j-1-1]
				w.Set(j-1, tmp1)
				(*iwork)[indibl+j-1-1] = itmp1
				goblas.Dswap(n, z.Vector(0, i-1), func() *int { y := 1; return &y }(), z.Vector(0, j-1), func() *int { y := 1; return &y }())
				if (*info) != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}
}
