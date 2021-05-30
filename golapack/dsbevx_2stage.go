package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dsbevx2stage computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric band matrix A using the 2stage technique for
// the reduction to tridiagonal. Eigenvalues and eigenvectors can
// be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
func Dsbevx2stage(jobz, _range, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, q *mat.Matrix, ldq *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork, ifail *[]int, info *int) {
	var alleig, indeig, lower, lquery, test, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, ib, iinfo, imax, indd, inde, indee, indhous, indibl, indisp, indiwo, indwrk, iscale, itmp1, j, jj, lhtrd, llwork, lwmin, lwtrd, nsplit int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lower = uplo == 'L'
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
	} else if (*kd) < 0 {
		(*info) = -5
	} else if (*ldab) < (*kd)+1 {
		(*info) = -7
	} else if wantz && (*ldq) < maxint(1, *n) {
		(*info) = -9
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -11
			}
		} else if indeig {
			if (*il) < 1 || (*il) > maxint(1, *n) {
				(*info) = -12
			} else if (*iu) < minint(*n, *il) || (*iu) > (*n) {
				(*info) = -13
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -18
		}
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
			work.Set(0, float64(lwmin))
		} else {
			ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, toPtr(-1), toPtr(-1))
			lhtrd = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwtrd = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("DSYTRD_SB2ST"), []byte{jobz}, n, kd, &ib, toPtr(-1))
			lwmin = 2*(*n) + lhtrd + lwtrd
			work.Set(0, float64(lwmin))
		}

		if (*lwork) < lwmin && !lquery {
			(*info) = -20
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBEVX_2STAGE"), -(*info))
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
		(*m) = 1
		if lower {
			tmp1 = ab.Get(0, 0)
		} else {
			tmp1 = ab.Get((*kd)+1-1, 0)
		}
		if valeig {
			if !((*vl) < tmp1 && (*vu) >= tmp1) {
				(*m) = 0
			}
		}
		if (*m) == 1 {
			w.Set(0, tmp1)
			if wantz {
				z.Set(0, 0, one)
			}
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
	anrm = Dlansb('M', uplo, n, kd, ab, ldab, work)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if lower {
			Dlascl('B', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		} else {
			Dlascl('Q', kd, kd, &one, &sigma, n, n, ab, ldab, info)
		}
		if (*abstol) > 0 {
			abstll = (*abstol) * sigma
		}
		if valeig {
			vll = (*vl) * sigma
			vuu = (*vu) * sigma
		}
	}

	//     Call DSYTRD_SB2ST to reduce symmetric band matrix to tridiagonal form.
	indd = 1
	inde = indd + (*n)
	indhous = inde + (*n)
	indwrk = indhous + lhtrd
	llwork = (*lwork) - indwrk + 1

	DsytrdSb2st('N', jobz, uplo, n, kd, ab, ldab, work.Off(indd-1), work.Off(inde-1), work.Off(indhous-1), &lhtrd, work.Off(indwrk-1), &llwork, &iinfo)

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
		if !wantz {
			goblas.Dcopy(toPtr((*n)-1), work.Off(inde-1), func() *int { y := 1; return &y }(), work.Off(indee-1), func() *int { y := 1; return &y }())
			Dsterf(n, w, work.Off(indee-1), info)
		} else {
			Dlacpy('A', n, n, q, ldq, z, ldz)
			goblas.Dcopy(toPtr((*n)-1), work.Off(inde-1), func() *int { y := 1; return &y }(), work.Off(indee-1), func() *int { y := 1; return &y }())
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
		for j = 1; j <= (*m); j++ {
			goblas.Dcopy(n, z.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
			goblas.Dgemv(NoTrans, n, n, &one, q, ldq, work, func() *int { y := 1; return &y }(), &zero, z.Vector(0, j-1), func() *int { y := 1; return &y }())
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label30:
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
				goblas.Dswap(n, z.Vector(0, i-1), func() *int { y := 1; return &y }(), z.Vector(0, j-1), func() *int { y := 1; return &y }())
				if (*info) != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwmin))
}