package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dstev computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric tridiagonal matrix A.
func Dstev(jobz byte, n *int, d, e *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var wantz bool
	var bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tnrm, zero float64
	var imax, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -6
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSTEV "), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
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
	rmax = math.Sqrt(bignum)

	//     Scale matrix to allowable range, if necessary.
	iscale = 0
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
	}

	//     For eigenvalues only, call DSTERF.  For eigenvalues and
	//     eigenvectors, call DSTEQR.
	if !wantz {
		Dsterf(n, d, e, info)
	} else {
		Dsteqr('I', n, d, e, z, ldz, work, info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*n)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(&imax, toPtrf64(one/sigma), d, toPtr(1))
	}
}
