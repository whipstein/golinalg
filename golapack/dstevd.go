package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstevd computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric tridiagonal matrix. If eigenvectors are desired, it
// uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dstevd(jobz byte, n *int, d, e *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, wantz bool
	var bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tnrm, zero float64
	var iscale, liwmin, lwmin int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	lquery = ((*lwork) == -1 || (*liwork) == -1)

	(*info) = 0
	liwmin = 1
	lwmin = 1
	if (*n) > 1 && wantz {
		lwmin = 1 + 4*(*n) + int(math.Pow(float64(*n), 2))
		liwmin = 3 + 5*(*n)
	}

	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -6
	}

	if (*info) == 0 {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -10
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSTEVD"), -(*info))
		return
	} else if lquery {
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
		goblas.Dscal(*n, sigma, d, 1)
		goblas.Dscal((*n)-1, sigma, e, 1)
	}

	//     For eigenvalues only, call DSTERF.  For eigenvalues and
	//     eigenvectors, call DSTEDC.
	if !wantz {
		Dsterf(n, d, e, info)
	} else {
		Dstedc('I', n, d, e, z, ldz, work, lwork, iwork, liwork, info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		goblas.Dscal(*n, one/sigma, d, 1)
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
