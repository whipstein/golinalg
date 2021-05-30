package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dsyevx computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A.  Eigenvalues and eigenvectors can be
// selected by specifying either a _range of values or a _range of indices
// for the desired eigenvalues.
func Dsyevx(jobz, _range, uplo byte, n *int, a *mat.Matrix, lda *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork, ifail *[]int, info *int) {
	var alleig, indeig, lower, lquery, test, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwo, indtau, indwkn, indwrk, iscale, itmp1, j, jj, llwork, llwrkn, lwkmin, lwkopt, nb, nsplit int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	lower = uplo == 'L'
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lquery = ((*lwork) == -1)

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(alleig || valeig || indeig) {
		(*info) = -2
	} else if !(lower || uplo == 'U') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -8
			}
		} else if indeig {
			if (*il) < 1 || (*il) > maxint(1, *n) {
				(*info) = -9
			} else if (*iu) < minint(*n, *il) || (*iu) > (*n) {
				(*info) = -10
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -15
		}
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwkmin = 1
			work.Set(0, float64(lwkmin))
		} else {
			lwkmin = 8 * (*n)
			nb = Ilaenv(toPtr(1), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
			nb = maxint(nb, Ilaenv(toPtr(1), []byte("DORMTR"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
			lwkopt = maxint(lwkmin, (nb+3)*(*n))
			work.Set(0, float64(lwkopt))
		}

		if (*lwork) < lwkmin && !lquery {
			(*info) = -17
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYEVX"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	(*m) = 0
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if alleig || indeig {
			(*m) = 1
			w.Set(0, a.Get(0, 0))
		} else {
			if (*vl) < a.Get(0, 0) && (*vu) >= a.Get(0, 0) {
				(*m) = 1
				w.Set(0, a.Get(0, 0))
			}
		}
		if wantz {
			z.Set(0, 0, one)
		}
		return
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = minf64(math.Sqrt(bignum), one/math.Sqrt(math.Sqrt(safmin)))

	//     Scale matrix to allowable _range, if necessary.
	iscale = 0
	abstll = (*abstol)
	if valeig {
		vll = (*vl)
		vuu = (*vu)
	}
	anrm = Dlansy('M', uplo, n, a, lda, work)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if lower {
			for j = 1; j <= (*n); j++ {
				goblas.Dscal(toPtr((*n)-j+1), &sigma, a.Vector(j-1, j-1), toPtr(1))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				goblas.Dscal(&j, &sigma, a.Vector(0, j-1), toPtr(1))
			}
		}
		if (*abstol) > 0 {
			abstll = (*abstol) * sigma
		}
		if valeig {
			vll = (*vl) * sigma
			vuu = (*vu) * sigma
		}
	}

	//     Call DSYTRD to reduce symmetric matrix to tridiagonal form.
	indtau = 1
	inde = indtau + (*n)
	indd = inde + (*n)
	indwrk = indd + (*n)
	llwork = (*lwork) - indwrk + 1
	Dsytrd(uplo, n, a, lda, work.Off(indd-1), work.Off(inde-1), work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)

	//     If all eigenvalues are desired and ABSTOL is less than or equal to
	//     zero, then call DSTERF or DORGTR and SSTEQR.  If this fails for
	//     some eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ((*abstol) <= zero) {
		goblas.Dcopy(n, work.Off(indd-1), toPtr(1), w, toPtr(1))
		indee = indwrk + 2*(*n)
		if !wantz {
			goblas.Dcopy(toPtr((*n)-1), work.Off(inde-1), toPtr(1), work.Off(indee-1), toPtr(1))
			Dsterf(n, w, work.Off(indee-1), info)
		} else {
			Dlacpy('A', n, n, a, lda, z, ldz)
			Dorgtr(uplo, n, z, ldz, work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)
			goblas.Dcopy(toPtr((*n)-1), work.Off(inde-1), toPtr(1), work.Off(indee-1), toPtr(1))
			Dsteqr(jobz, n, w, work.Off(indee-1), z, ldz, work.Off(indwrk-1), info)
			if (*info) == 0 {
				for i = 1; i <= (*n); i++ {
					(*ifail)[i-1] = 0
				}
			}
		}
		if (*info) == 0 {
			(*m) = (*n)
			goto label40
		}
		(*info) = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, SSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + (*n)
	indiwo = indisp + (*n)
	Dstebz(_range, order, n, &vll, &vuu, il, iu, &abstll, work.Off(indd-1), work.Off(inde-1), m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwrk-1), toSlice(iwork, indiwo-1), info)

	if wantz {
		Dstein(n, work.Off(indd-1), work.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, work.Off(indwrk-1), toSlice(iwork, indiwo-1), ifail, info)

		//        Apply orthogonal matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by DSTEIN.
		indwkn = inde
		llwrkn = (*lwork) - indwkn + 1
		Dormtr('L', uplo, 'N', n, m, a, lda, work.Off(indtau-1), z, ldz, work.Off(indwkn-1), &llwrkn, &iinfo)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label40:
	;
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*m)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(&imax, toPtrf64(one/sigma), w, toPtr(1))
	}

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
				goblas.Dswap(n, z.Vector(0, i-1), toPtr(1), z.Vector(0, j-1), toPtr(1))
				if (*info) != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwkopt))
}
