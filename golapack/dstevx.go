package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstevx computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric tridiagonal matrix A.  Eigenvalues and
// eigenvectors can be selected by specifying either a _range of values
// or a _range of indices for the desired eigenvalues.
func Dstevx(jobz, _range byte, n *int, d, e *mat.Vector, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, iwork, ifail *[]int, info *int) {
	var alleig, indeig, test, valeig, wantz bool
	var order byte
	var bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, tnrm, vll, vuu, zero float64
	var i, imax, indibl, indisp, indiwo, indwrk, iscale, itmp1, j, jj, nsplit int

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
	} else if (*n) < 0 {
		(*info) = -3
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
		gltest.Xerbla([]byte("DSTEVX"), -(*info))
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
			w.Set(0, d.Get(0))
		} else {
			if (*vl) < d.Get(0) && (*vu) >= d.Get(0) {
				(*m) = 1
				w.Set(0, d.Get(0))
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
	if valeig {
		vll = (*vl)
		vuu = (*vu)
	} else {
		vll = zero
		vuu = zero
	}
	tnrm = Dlanst('M', n, d, e)
	if tnrm > zero && tnrm < rmin {
		iscale = 1
		sigma = rmin / tnrm
	} else if tnrm > rmax {
		iscale = 1
		sigma = rmax / tnrm
	}
	if iscale == 1 {
		goblas.Dscal(n, &sigma, d, toPtr(1))
		goblas.Dscal(toPtr((*n)-1), &sigma, e, toPtr(1))
		if valeig {
			vll = (*vl) * sigma
			vuu = (*vu) * sigma
		}
	}

	//     If all eigenvalues are desired and ABSTOL is less than zero, then
	//     call DSTERF or SSTEQR.  If this fails for some eigenvalue, then
	//     try DSTEBZ.
	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ((*abstol) <= zero) {
		goblas.Dcopy(n, d, toPtr(1), w, toPtr(1))
		goblas.Dcopy(toPtr((*n)-1), e, toPtr(1), work, toPtr(1))
		indwrk = (*n) + 1
		if !wantz {
			Dsterf(n, w, work, info)
		} else {
			Dsteqr('I', n, w, work, z, ldz, work.Off(indwrk-1), info)
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
	indwrk = 1
	indibl = 1
	indisp = indibl + (*n)
	indiwo = indisp + (*n)
	Dstebz(_range, order, n, &vll, &vuu, il, iu, abstol, d, e, m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwrk-1), toSlice(iwork, indiwo-1), info)

	if wantz {
		Dstein(n, d, e, m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, work.Off(indwrk-1), toSlice(iwork, indiwo-1), ifail, info)
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
