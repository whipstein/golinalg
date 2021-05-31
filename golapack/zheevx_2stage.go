package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheevx2stage computes selected eigenvalues and, optionally, eigenvectors
// of a complex Hermitian matrix A using the 2stage technique for
// the reduction to tridiagonal.  Eigenvalues and eigenvectors can
// be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
func Zheevx2stage(jobz, _range, uplo byte, n *int, a *mat.CMatrix, lda *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, lower, lquery, test, valeig, wantz bool
	var order byte
	var cone complex128
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, ib, iinfo, imax, indd, inde, indee, indhous, indibl, indisp, indiwk, indrwk, indtau, indwrk, iscale, itmp1, j, jj, kd, lhtrd, llwork, lwmin, lwtrd, nsplit int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	lower = uplo == 'L'
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lquery = ((*lwork) == -1)

	(*info) = 0
	if jobz != 'N' {
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
			lwmin = 1
			work.SetRe(0, float64(lwmin))
		} else {
			kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, toPtr(-1), toPtr(-1), toPtr(-1))
			ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, toPtr(-1), toPtr(-1))
			lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
			lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_2STAGE"), []byte{jobz}, n, &kd, &ib, toPtr(-1))
			lwmin = (*n) + lhtrd + lwtrd
			work.SetRe(0, float64(lwmin))
		}
		//
		if (*lwork) < lwmin && !lquery {
			(*info) = -17
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHEEVX_2STAGE"), -(*info))
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
			w.Set(0, a.GetRe(0, 0))
		} else if valeig {
			if (*vl) < a.GetRe(0, 0) && (*vu) >= a.GetRe(0, 0) {
				(*m) = 1
				w.Set(0, a.GetRe(0, 0))
			}
		}
		if wantz {
			z.Set(0, 0, cone)
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
	anrm = Zlanhe('M', uplo, n, a, lda, rwork)
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
				goblas.Zdscal(toPtr((*n)-j+1), &sigma, a.CVector(j-1, j-1), func() *int { y := 1; return &y }())
			}
		} else {
			for j = 1; j <= (*n); j++ {
				goblas.Zdscal(&j, &sigma, a.CVector(0, j-1), func() *int { y := 1; return &y }())
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

	//     Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form.
	indd = 1
	inde = indd + (*n)
	indrwk = inde + (*n)
	indtau = 1
	indhous = indtau + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	Zhetrd2stage(jobz, uplo, n, a, lda, rwork.Off(indd-1), rwork.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

	//     If all eigenvalues are desired and ABSTOL is less than or equal to
	//     zero, then call DSTERF or ZUNGTR and ZSTEQR.  If this fails for
	//     some eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ((*abstol) <= zero) {
		goblas.Dcopy(n, rwork.Off(indd-1), func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }())
		indee = indrwk + 2*(*n)
		if !wantz {
			goblas.Dcopy(toPtr((*n)-1), rwork.Off(inde-1), func() *int { y := 1; return &y }(), rwork.Off(indee-1), func() *int { y := 1; return &y }())
			Dsterf(n, w, rwork.Off(indee-1), info)
		} else {
			Zlacpy('A', n, n, a, lda, z, ldz)
			Zungtr(uplo, n, z, ldz, work.Off(indtau-1), work.Off(indwrk-1), &llwork, &iinfo)
			goblas.Dcopy(toPtr((*n)-1), rwork.Off(inde-1), func() *int { y := 1; return &y }(), rwork.Off(indee-1), func() *int { y := 1; return &y }())
			Zsteqr(jobz, n, w, rwork.Off(indee-1), z, ldz, rwork.Off(indrwk-1), info)
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

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + (*n)
	indiwk = indisp + (*n)
	Dstebz(_range, order, n, &vll, &vuu, il, iu, &abstll, rwork.Off(indd-1), rwork.Off(inde-1), m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), rwork.Off(indrwk-1), toSlice(iwork, indiwk-1), info)

	if wantz {
		Zstein(n, rwork.Off(indd-1), rwork.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, rwork.Off(indrwk-1), toSlice(iwork, indiwk-1), ifail, info)

		//        Apply unitary matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by ZSTEIN.
		Zunmtr('L', uplo, 'N', n, m, a, lda, work.Off(indtau-1), z, ldz, work.Off(indwrk-1), &llwork, &iinfo)
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
		goblas.Dscal(&imax, toPtrf64(one/sigma), w, func() *int { y := 1; return &y }())
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
				goblas.Zswap(n, z.CVector(0, i-1), func() *int { y := 1; return &y }(), z.CVector(0, j-1), func() *int { y := 1; return &y }())
				if (*info) != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}

	//     Set WORK(1) to optimal complex workspace size.
	work.SetRe(0, float64(lwmin))
}
