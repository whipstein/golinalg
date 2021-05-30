package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dspevx computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A in packed storage.  Eigenvalues/vectors
// can be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
func Dspevx(jobz, _range, uplo byte, n *int, ap *mat.Vector, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, test, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwo, indtau, indwrk, iscale, itmp1, j, jj, nsplit int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(alleig || valeig || indeig) {
		(*info) = -2
	} else if !(uplo == 'L' || uplo == 'U') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -7
			}
		} else if indeig {
			if (*il) < 1 || (*il) > maxint(1, *n) {
				(*info) = -8
			} else if (*iu) < minint(*n, *il) || (*iu) > (*n) {
				(*info) = -9
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -14
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPEVX"), -(*info))
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
			w.Set(0, ap.Get(0))
		} else {
			if (*vl) < ap.Get(0) && (*vu) >= ap.Get(0) {
				(*m) = 1
				w.Set(0, ap.Get(0))
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
	} else {
		vll = zero
		vuu = zero
	}
	anrm = Dlansp('M', uplo, n, ap, work)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		goblas.Dscal(toPtr(((*n)*((*n)+1))/2), &sigma, ap, toPtr(1))
		if (*abstol) > 0 {
			abstll = (*abstol) * sigma
		}
		if valeig {
			vll = (*vl) * sigma
			vuu = (*vu) * sigma
		}
	}

	//     Call DSPTRD to reduce symmetric packed matrix to tridiagonal form.
	indtau = 1
	inde = indtau + (*n)
	indd = inde + (*n)
	indwrk = indd + (*n)
	Dsptrd(uplo, n, ap, work.Off(indd-1), work.Off(inde-1), work.Off(indtau-1), &iinfo)

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or DOPGTR and SSTEQR.  If this fails
	//     for some eigenvalue, then try DSTEBZ.
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
			Dopgtr(uplo, n, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
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
			goto label20
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
		Dopmtr('L', uplo, 'N', n, m, ap, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &iinfo)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label20:
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
}
