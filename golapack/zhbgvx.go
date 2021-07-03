package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbgvx computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
// and banded, and B is also positive definite.  Eigenvalues and
// eigenvectors can be selected by specifying either all eigenvalues,
// a _range of values or a _range of indices for the desired eigenvalues.
func Zhbgvx(jobz, _range, uplo byte, n, ka, kb *int, ab *mat.CMatrix, ldab *int, bb *mat.CMatrix, ldbb *int, q *mat.CMatrix, ldq *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, rwork *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, test, upper, valeig, wantz bool
	var order, vect byte
	var cone, czero complex128
	var tmp1, zero float64
	var i, iinfo, indd, inde, indee, indibl, indisp, indiwk, indrwk, indwrk, itmp1, j, jj, nsplit int
	var err error
	_ = err

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZHBGVX"), -(*info))
		return
	}

	//     Quick return if possible
	(*m) = 0
	if (*n) == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	Zpbstf(uplo, n, kb, bb, ldbb, info)
	if (*info) != 0 {
		(*info) = (*n) + (*info)
		return
	}

	//     Transform problem to standard eigenvalue problem.
	Zhbgst(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, work, rwork, &iinfo)

	//     Solve the standard eigenvalue problem.
	//     Reduce Hermitian band matrix to tridiagonal form.
	indd = 1
	inde = indd + (*n)
	indrwk = inde + (*n)
	indwrk = 1
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	Zhbtrd(vect, uplo, n, ka, ab, ldab, rwork.Off(indd-1), rwork.Off(inde-1), q, ldq, work.Off(indwrk-1), &iinfo)

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or ZSTEQR.  If this fails for some
	//     eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ((*abstol) <= zero) {
		goblas.Dcopy(*n, rwork.Off(indd-1), 1, w, 1)
		indee = indrwk + 2*(*n)
		goblas.Dcopy((*n)-1, rwork.Off(inde-1), 1, rwork.Off(indee-1), 1)
		if !wantz {
			Dsterf(n, w, rwork.Off(indee-1), info)
		} else {
			Zlacpy('A', n, n, q, ldq, z, ldz)
			Zsteqr(jobz, n, w, rwork.Off(indee-1), z, ldz, rwork.Off(indrwk-1), info)
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
	//     call ZSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + (*n)
	indiwk = indisp + (*n)
	Dstebz(_range, order, n, vl, vu, il, iu, abstol, rwork.Off(indd-1), rwork.Off(inde-1), m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), rwork.Off(indrwk-1), toSlice(iwork, indiwk-1), info)

	if wantz {
		Zstein(n, rwork.Off(indd-1), rwork.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, rwork.Off(indrwk-1), toSlice(iwork, indiwk-1), ifail, info)

		//        Apply unitary matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by ZSTEIN.
		for j = 1; j <= (*m); j++ {
			goblas.Zcopy(*n, z.CVector(0, j-1), 1, work.Off(0), 1)
			err = goblas.Zgemv(NoTrans, *n, *n, cone, q, *ldq, work, 1, czero, z.CVector(0, j-1), 1)
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
				goblas.Zswap(*n, z.CVector(0, i-1), 1, z.CVector(0, j-1), 1)
				if (*info) != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}
}
