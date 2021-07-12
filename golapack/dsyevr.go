package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyevr computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A.  Eigenvalues and eigenvectors can be
// selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
//
// DSYEVR first reduces the matrix A to tridiagonal form T with a call
// to DSYTRD.  Then, whenever possible, DSYEVR calls DSTEMR to compute
// the eigenspectrum using Relatively Robust Representations.  DSTEMR
// computes eigenvalues by the dqds algorithm, while orthogonal
// eigenvectors are computed from various "good" L D L^T representations
// (also known as Relatively Robust Representations). Gram-Schmidt
// orthogonalization is avoided as far as possible. More specifically,
// the various steps of the algorithm are as follows.
//
// For each unreduced block (submatrix) of T,
//    (a) Compute T - sigma I  = L D L^T, so that L and D
//        define all the wanted eigenvalues to high relative accuracy.
//        This means that small relative changes in the entries of D and L
//        cause only small relative changes in the eigenvalues and
//        eigenvectors. The standard (unfactored) representation of the
//        tridiagonal matrix T does not have this property in general.
//    (b) Compute the eigenvalues to suitable accuracy.
//        If the eigenvectors are desired, the algorithm attains full
//        accuracy of the computed eigenvalues only right before
//        the corresponding vectors have to be computed, see steps c) and d).
//    (c) For each cluster of close eigenvalues, select a new
//        shift close to the cluster, find a new factorization, and refine
//        the shifted eigenvalues to suitable accuracy.
//    (d) For each eigenvalue with a large enough relative separation compute
//        the corresponding eigenvector by forming a rank revealing twisted
//        factorization. Go back to (c) for any clusters that remain.
//
// The desired accuracy of the output can be specified by the input
// parameter ABSTOL.
//
// For more details, see DSTEMR's documentation and:
// - Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations
//   to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
//   Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
// - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
//   Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
//   2004.  Also LAPACK Working Note 154.
// - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
//   tridiagonal eigenvalue/eigenvector problem",
//   Computer Science Division Technical Report No. UCB/CSD-97-971,
//   UC Berkeley, May 1997.
//
//
// Note 1 : DSYEVR calls DSTEMR when the full spectrum is requested
// on machines which conform to the ieee-754 floating point standard.
// DSYEVR calls DSTEBZ and DSTEIN on non-ieee machines and
// when partial spectrum requests are made.
//
// Normal execution of DSTEMR may create NaNs and infinities and
// hence may abort due to a floating point exception in environments
// which do not handle NaNs and infinities in the ieee standard default
// manner.
func Dsyevr(jobz, _range, uplo byte, n *int, a *mat.Matrix, lda *int, vl, vu *float64, il, iu *int, abstol *float64, m *int, w *mat.Vector, z *mat.Matrix, ldz *int, isuppz *[]int, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var alleig, indeig, lower, lquery, tryrac, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, two, vll, vuu, zero float64
	var i, ieeeok, iinfo, imax, indd, inddd, inde, indee, indibl, indifl, indisp, indiwo, indtau, indwk, indwkn, iscale, j, jj, liwmin, llwork, llwrkn, lwkopt, lwmin, nb, nsplit int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	ieeeok = Ilaenv(func() *int { y := 10; return &y }(), []byte("DSYEVR"), []byte("N"), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 4; return &y }())

	lower = uplo == 'L'
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	lquery = (((*lwork) == -1) || ((*liwork) == -1))

	lwmin = max(1, 26*(*n))
	liwmin = max(1, 10*(*n))

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(alleig || valeig || indeig) {
		(*info) = -2
	} else if !(lower || uplo == 'U') {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *n) {
		(*info) = -6
	} else {
		if valeig {
			if (*n) > 0 && (*vu) <= (*vl) {
				(*info) = -8
			}
		} else if indeig {
			if (*il) < 1 || (*il) > max(1, *n) {
				(*info) = -9
			} else if (*iu) < min(*n, *il) || (*iu) > (*n) {
				(*info) = -10
			}
		}
	}
	if (*info) == 0 {
		if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
			(*info) = -15
		} else if (*lwork) < lwmin && !lquery {
			(*info) = -18
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -20
		}
	}

	if (*info) == 0 {
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		nb = max(nb, Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMTR"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
		lwkopt = max((nb+1)*(*n), lwmin)
		work.Set(0, float64(lwkopt))
		(*iwork)[0] = liwmin
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYEVR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	(*m) = 0
	if (*n) == 0 {
		work.Set(0, 1)
		return
	}

	if (*n) == 1 {
		work.Set(0, 7)
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
			(*isuppz)[0] = 1
			(*isuppz)[1] = 1
		}
		return
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = math.Min(math.Sqrt(bignum), one/math.Sqrt(math.Sqrt(safmin)))

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
				goblas.Dscal((*n)-j+1, sigma, a.Vector(j-1, j-1, 1))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				goblas.Dscal(j, sigma, a.Vector(0, j-1, 1))
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
	//     Initialize indices into workspaces.  Note: The IWORK indices are
	//     used only if DSTERF or DSTEMR fail.
	//     WORK(INDTAU:INDTAU+N-1) stores the scalar factors of the
	//     elementary reflectors used in DSYTRD.
	indtau = 1
	//     WORK(INDD:INDD+N-1) stores the tridiagonal's diagonal entries.
	indd = indtau + (*n)
	//     WORK(INDE:INDE+N-1) stores the off-diagonal entries of the
	//     tridiagonal matrix from DSYTRD.
	inde = indd + (*n)
	//     WORK(INDDD:INDDD+N-1) is a copy of the diagonal entries over
	//     -written by DSTEMR (the DSTERF path copies the diagonal to W).
	inddd = inde + (*n)
	//     WORK(INDEE:INDEE+N-1) is a copy of the off-diagonal entries over
	//     -written while computing the eigenvalues in DSTERF and DSTEMR.
	indee = inddd + (*n)
	//     INDWK is the starting offset of the left-over workspace, and
	//     LLWORK is the remaining workspace size.
	indwk = indee + (*n)
	llwork = (*lwork) - indwk + 1
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
	indiwo = indifl + (*n)

	//     Call DSYTRD to reduce symmetric matrix to tridiagonal form.
	Dsytrd(uplo, n, a, lda, work.Off(indd-1), work.Off(inde-1), work.Off(indtau-1), work.Off(indwk-1), &llwork, &iinfo)

	//     If all eigenvalues are desired
	//     then call DSTERF or DSTEMR and DORMTR.
	if (alleig || (indeig && (*il) == 1 && (*iu) == (*n))) && ieeeok == 1 {
		if !wantz {
			goblas.Dcopy(*n, work.Off(indd-1, 1), w.Off(0, 1))
			goblas.Dcopy((*n)-1, work.Off(inde-1, 1), work.Off(indee-1, 1))
			Dsterf(n, w, work.Off(indee-1), info)
		} else {
			goblas.Dcopy((*n)-1, work.Off(inde-1, 1), work.Off(indee-1, 1))
			goblas.Dcopy(*n, work.Off(indd-1, 1), work.Off(inddd-1, 1))

			if (*abstol) <= two*float64(*n)*eps {
				tryrac = true
			} else {
				tryrac = false
			}
			Dstemr(jobz, 'A', n, work.Off(inddd-1), work.Off(indee-1), vl, vu, il, iu, m, w, z, ldz, n, isuppz, &tryrac, work.Off(indwk-1), lwork, iwork, liwork, info)

			//        Apply orthogonal matrix used in reduction to tridiagonal
			//        form to eigenvectors returned by DSTEMR.
			if wantz && (*info) == 0 {
				indwkn = inde
				llwrkn = (*lwork) - indwkn + 1
				Dormtr('L', uplo, 'N', n, m, a, lda, work.Off(indtau-1), z, ldz, work.Off(indwkn-1), &llwrkn, &iinfo)
			}
		}

		if (*info) == 0 {
			//           Everything worked.  Skip DSTEBZ/DSTEIN.  IWORK(:) are
			//           undefined.
			(*m) = (*n)
			goto label30
		}
		(*info) = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
	//     Also call DSTEBZ and DSTEIN if DSTEMR fails.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	Dstebz(_range, order, n, &vll, &vuu, il, iu, &abstll, work.Off(indd-1), work.Off(inde-1), m, &nsplit, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwk-1), toSlice(iwork, indiwo-1), info)

	if wantz {
		Dstein(n, work.Off(indd-1), work.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, ldz, work.Off(indwk-1), toSlice(iwork, indiwo-1), toSlice(iwork, indifl-1), info)

		//        Apply orthogonal matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by DSTEIN.
		indwkn = inde
		llwrkn = (*lwork) - indwkn + 1
		Dormtr('L', uplo, 'N', n, m, a, lda, work.Off(indtau-1), z, ldz, work.Off(indwkn-1), &llwrkn, &iinfo)
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	//
	//  Jump here if DSTEMR/DSTEIN succeeded.
label30:
	;
	if iscale == 1 {
		if (*info) == 0 {
			imax = (*m)
		} else {
			imax = (*info) - 1
		}
		goblas.Dscal(imax, one/sigma, w.Off(0, 1))
	}

	//     If eigenvalues are not in order, then sort them, along with
	//     eigenvectors.  Note: We do not sort the IFAIL portion of IWORK.
	//     It may not be initialized (if DSTEMR/DSTEIN succeeded), and we do
	//     not return this detailed information to the user.
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
				w.Set(i-1, w.Get(j-1))
				w.Set(j-1, tmp1)
				goblas.Dswap(*n, z.Vector(0, i-1, 1), z.Vector(0, j-1, 1))
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwkopt))
	(*iwork)[0] = liwmin
}
