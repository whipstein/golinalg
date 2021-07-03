package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstevr computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric tridiagonal matrix T.  Eigenvalues and
// eigenvectors can be selected by specifying either a _range of values
// or a _range of indices for the desired eigenvalues.
//
// Whenever possible, DSTEVR calls DSTEMR to compute the
// eigenspectrum using Relatively Robust Representations.  DSTEMR
// computes eigenvalues by the dqds algorithm, while orthogonal
// eigenvectors are computed from various "good" L D L^T representations
// (also known as Relatively Robust Representations). Gram-Schmidt
// orthogonalization is avoided as far as possible. More specifically,
// the various steps of the algorithm are as follows. For the i-th
// unreduced block of T,
//    (a) Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
//         is a relatively robust representation,
//    (b) Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
//        relative accuracy by the dqds algorithm,
//    (c) If there is a cluster of close eigenvalues, "choose" sigma_i
//        close to the cluster, and go to step (a),
//    (d) Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
//        compute the corresponding eigenvector by forming a
//        rank-revealing twisted factorization.
// The desired accuracy of the output can be specified by the input
// parameter ABSTOL.
//
// For more details, see "A new O(n^2) algorithm for the symmetric
// tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
// Computer Science Division Technical Report No. UCB//CSD-97-971,
// UC Berkeley, May 1997.
//
//
// Note 1 : DSTEVR calls DSTEMR when the full spectrum is requested
// on machines which conform to the ieee-754 floating point standard.
// DSTEVR calls DSTEBZ and DSTEIN on non-ieee machines and
// when partial spectrum requests are made.
//
// Normal execution of DSTEMR may create NaNs and infinities and
// hence may abort due to a floating point exception in environments
// which do not handle NaNs and infinities in the ieee standard default
// manner.
func Dstevr(jobz, _range byte, n *int, d, e *mat.Vector, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, isuppz *[]int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var alleig, indeig, lquery, test, tryrac, valeig, wantz bool
	var order byte
	var bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, tnrm, two, vll, vuu, zero float64
	var i, ieeeok, imax, indibl, indifl, indisp, indiwo, iscale, itmp1, j, jj, liwmin, lwmin, nsplit int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	ieeeok = Ilaenv(func() *int { y := 10; return &y }(), []byte("DSTEVR"), []byte("N"), toPtr(1), toPtr(2), toPtr(3), toPtr(4))

	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	lquery = (((*lwork) == -1) || ((*liwork) == -1))
	lwmin = maxint(1, 20*(*n))
	liwmin = maxint(1, 10*(*n))

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

	if (*info) == 0 {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -17
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -19
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSTEVR"), -(*info))
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
		goblas.Dscal(*n, sigma, d, 1)
		goblas.Dscal((*n)-1, sigma, e, 1)
		if valeig {
			vll = (*vl) * sigma
			vuu = (*vu) * sigma
		}
	}
	//     Initialize indices into workspaces.  Note: These indices are used only
	//     if DSTERF or DSTEMR fail.
	//     IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
	//     stores the block indices of each of the M<=N eigenvalues.
	indibl = 1
	//     IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
	//     stores the starting and finishing indices of each block.
	indisp = indibl + (*n)
	//     IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
	//     that corresponding to eigenvectors that fail to converge in
	//     DSTEIN.  This information is discarded; if any fail, the driver
	//     returns INFO > 0.
	indifl = indisp + (*n)
	//     INDIWO is the offset of the remaining integer workspace.
	indiwo = indisp + (*n)
	//
	//     If all eigenvalues are desired, then
	//     call DSTERF or DSTEMR.  If this fails for some eigenvalue, then
	//     try DSTEBZ.

	test = false
	if indeig {
		if (*il) == 1 && (*iu) == (*n) {
			test = true
		}
	}
	if (alleig || test) && ieeeok == 1 {
		goblas.Dcopy((*n)-1, e, 1, work, 1)
		if !wantz {
			goblas.Dcopy(*n, d, 1, w, 1)
			Dsterf(n, w, work, info)
		} else {
			goblas.Dcopy(*n, d, 1, work.Off((*n)+1-1), 1)
			if (*abstol) <= two*float64(*n)*eps {
				tryrac = true
			} else {
				tryrac = false
			}
			Dstemr(jobz, 'A', n, work.Off((*n)+1-1), work, vl, vu, il, iu, m, w, z, ldz, n, isuppz, &tryrac, work.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), iwork, liwork, info)

		}
		if (*info) == 0 {
			(*m) = (*n)
			goto label10
		}
		(*info) = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	Dstebz(_range, order, n, &vll, &vuu, il, iu, abstol, d, e, m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work, toSlice(iwork, indiwo-1), info)

	if wantz {
		Dstein(n, d, e, m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, work, toSlice(iwork, indiwo-1), toSlice(iwork, indifl-1), info)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label10:
	;
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*m)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(imax, one/sigma, w, 1)
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
				itmp1 = (*iwork)[i-1]
				w.Set(i-1, w.Get(j-1))
				(*iwork)[i-1] = (*iwork)[j-1]
				w.Set(j-1, tmp1)
				(*iwork)[j-1] = itmp1
				goblas.Dswap(*n, z.Vector(0, i-1), 1, z.Vector(0, j-1), 1)
			}
		}
	}

	//      Causes problems with tests 19 & 20:
	//      IF (wantz .and. INDEIG ) Z( 1,1) = Z(1,1) / 1.002 + .002
	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
