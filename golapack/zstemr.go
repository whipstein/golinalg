package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zstemr computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric tridiagonal matrix T. Any such unreduced matrix has
// a well defined set of pairwise different real eigenvalues, the corresponding
// real eigenvectors are pairwise orthogonal.
//
// The spectrum may be computed either completely or partially by specifying
// either an interval (VL,VU] or a _range of indices IL:IU for the desired
// eigenvalues.
//
// Depending on the number of desired eigenvalues, these are computed either
// by bisection or the dqds algorithm. Numerically orthogonal eigenvectors are
// computed by the use of various suitable L D L^T factorizations near clusters
// of close eigenvalues (referred to as RRRs, Relatively Robust
// Representations). An informal sketch of the algorithm follows.
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
// For more details, see:
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
// Further Details
// 1.ZSTEMR works only on machines which follow IEEE-754
// floating-point standard in their handling of infinities and NaNs.
// This permits the use of efficient inner loops avoiding a check for
// zero divisors.
//
// 2. LAPACK routines can be used to reduce a complex Hermitean matrix to
// real symmetric tridiagonal form.
//
// (Any complex Hermitean tridiagonal matrix has real values on its diagonal
// and potentially complex numbers on its off-diagonals. By applying a
// similarity transform with an appropriate diagonal matrix
// diag(1,e^{i \phy_1}, ... , e^{i \phy_{n-1}}), the complex Hermitean
// matrix can be transformed into a real symmetric matrix and complex
// arithmetic can be entirely avoided.)
//
// While the eigenvectors of the real symmetric tridiagonal matrix are real,
// the eigenvectors of original complex Hermitean matrix have complex entries
// in general.
// Since LAPACK drivers overwrite the matrix data with the eigenvectors,
// ZSTEMR accepts complex workspace to facilitate interoperability
// with ZUNMTR or ZUPMTR.
func Zstemr(jobz, _range byte, n *int, d, e *mat.Vector, vl, vu *float64, il, iu, m *int, w *mat.Vector, z *mat.CMatrix, ldz, nzc *int, isuppz *[]int, tryrac *bool, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var alleig, indeig, lquery, valeig, wantz, zquery bool
	var bignum, cs, eps, four, minrgp, one, pivmin, r1, r2, rmax, rmin, rtol1, rtol2, safmin, scale, smlnum, sn, thresh, tmp, tnrm, wl, wu, zero float64
	var i, ibegin, iend, ifirst, iil, iindbl, iindw, iindwk, iinfo, iinspl, iiu, ilast, in, indd, inde2, inderr, indgp, indgrs, indwrk, itmp, itmp2, j, jblk, jj, liwmin, lwmin, nsplit, nzcmin, offset, wbegin, wend int

	zero = 0.0
	one = 1.0
	four = 4.0
	minrgp = 1.0e-3

	//     Test the input parameters.
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	lquery = (((*lwork) == -1) || ((*liwork) == -1))
	zquery = ((*nzc) == -1)
	//     DSTEMR needs WORK of size 6*N, IWORK of size 3*N.
	//     In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N.
	//     Furthermore, ZLARRV needs WORK of size 12*N, IWORK of size 7*N.
	if wantz {
		lwmin = 18 * (*n)
		liwmin = 10 * (*n)
	} else {
		//        need less workspace if only the eigenvalues are wanted
		lwmin = 12 * (*n)
		liwmin = 8 * (*n)
	}
	wl = zero
	wu = zero
	iil = 0
	iiu = 0
	nsplit = 0
	if valeig {
		//        We do not reference VL, VU in the cases RANGE = 'I','A'
		//        The interval (WL, WU] contains all the wanted eigenvalues.
		//        It is either given by the user or computed in DLARRE.
		wl = (*vl)
		wu = (*vu)
	} else if indeig {
		//        We do not reference IL, IU in the cases RANGE = 'V','A'
		iil = (*il)
		iiu = (*iu)
	}

	(*info) = 0
	if !(wantz || jobz == 'N') {
		(*info) = -1
	} else if !(alleig || valeig || indeig) {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if valeig && (*n) > 0 && wu <= wl {
		(*info) = -7
	} else if indeig && (iil < 1 || iil > (*n)) {
		(*info) = -8
	} else if indeig && (iiu < iil || iiu > (*n)) {
		(*info) = -9
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -13
	} else if (*lwork) < lwmin && !lquery {
		(*info) = -17
	} else if (*liwork) < liwmin && !lquery {
		(*info) = -19
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = minf64(math.Sqrt(bignum), one/math.Sqrt(math.Sqrt(safmin)))

	if (*info) == 0 {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if wantz && alleig {
			nzcmin = (*n)
		} else if wantz && valeig {
			Dlarrc('T', n, vl, vu, d, e, &safmin, &nzcmin, &itmp, &itmp2, info)
		} else if wantz && indeig {
			nzcmin = iiu - iil + 1
		} else {
			//           WANTZ .EQ. FALSE.
			nzcmin = 0
		}
		if zquery && (*info) == 0 {
			z.SetRe(0, 0, float64(nzcmin))
		} else if (*nzc) < nzcmin && !zquery {
			(*info) = -14
		}
	}
	if (*info) != 0 {

		gltest.Xerbla([]byte("ZSTEMR"), -(*info))

		return
	} else if lquery || zquery {
		return
	}

	//     Handle N = 0, 1, and 2 cases immediately
	(*m) = 0
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if alleig || indeig {
			(*m) = 1
			w.Set(0, d.Get(0))
		} else {
			if wl < d.Get(0) && wu >= d.Get(0) {
				(*m) = 1
				w.Set(0, d.Get(0))
			}
		}
		if wantz && (!zquery) {
			z.SetRe(0, 0, one)
			(*isuppz)[0] = 1
			(*isuppz)[1] = 1
		}
		return
	}

	if (*n) == 2 {
		if !wantz {
			Dlae2(d.GetPtr(0), e.GetPtr(0), d.GetPtr(1), &r1, &r2)
		} else if wantz && (!zquery) {
			Dlaev2(d.GetPtr(0), e.GetPtr(0), d.GetPtr(1), &r1, &r2, &cs, &sn)
		}
		if alleig || (valeig && (r2 > wl) && (r2 <= wu)) || (indeig && (iil == 1)) {
			(*m) = (*m) + 1
			w.Set((*m)-1, r2)
			if wantz && (!zquery) {
				z.SetRe(0, (*m)-1, -sn)
				z.SetRe(1, (*m)-1, cs)
				//              Note: At most one of SN and CS can be zero.
				if sn != zero {
					if cs != zero {
						(*isuppz)[2*(*m)-1-1] = 1
						(*isuppz)[2*(*m)-1] = 2
					} else {
						(*isuppz)[2*(*m)-1-1] = 1
						(*isuppz)[2*(*m)-1] = 1
					}
				} else {
					(*isuppz)[2*(*m)-1-1] = 2
					(*isuppz)[2*(*m)-1] = 2
				}
			}
		}
		if alleig || (valeig && (r1 > wl) && (r1 <= wu)) || (indeig && (iiu == 2)) {
			(*m) = (*m) + 1
			w.Set((*m)-1, r1)
			if wantz && (!zquery) {
				z.SetRe(0, (*m)-1, cs)
				z.SetRe(1, (*m)-1, sn)
				//              Note: At most one of SN and CS can be zero.
				if sn != zero {
					if cs != zero {
						(*isuppz)[2*(*m)-1-1] = 1
						(*isuppz)[2*(*m)-1] = 2
					} else {
						(*isuppz)[2*(*m)-1-1] = 1
						(*isuppz)[2*(*m)-1] = 1
					}
				} else {
					(*isuppz)[2*(*m)-1-1] = 2
					(*isuppz)[2*(*m)-1] = 2
				}
			}
		}
	} else {
		//        Continue with general N
		indgrs = 1
		inderr = 2*(*n) + 1
		indgp = 3*(*n) + 1
		indd = 4*(*n) + 1
		inde2 = 5*(*n) + 1
		indwrk = 6*(*n) + 1

		iinspl = 1
		iindbl = (*n) + 1
		iindw = 2*(*n) + 1
		iindwk = 3*(*n) + 1

		//        Scale matrix to allowable _range, if necessary.
		//        The allowable _range is related to the PIVMIN parameter; see the
		//        comments in DLARRD.  The preference for scaling small values
		//        up is heuristic; we expect users' matrices not to be close to the
		//        RMAX threshold.
		scale = one
		tnrm = Dlanst('M', n, d, e)
		if tnrm > zero && tnrm < rmin {
			scale = rmin / tnrm
		} else if tnrm > rmax {
			scale = rmax / tnrm
		}
		if scale != one {
			goblas.Dscal(n, &scale, d, func() *int { y := 1; return &y }())
			goblas.Dscal(toPtr((*n)-1), &scale, e, func() *int { y := 1; return &y }())
			tnrm = tnrm * scale
			if valeig {
				//              If eigenvalues in interval have to be found,
				//              scale (WL, WU] accordingly
				wl = wl * scale
				wu = wu * scale
			}
		}

		//        Compute the desired eigenvalues of the tridiagonal after splitting
		//        into smaller subblocks if the corresponding off-diagonal elements
		//        are small
		//        THRESH is the splitting parameter for DLARRE
		//        A negative THRESH forces the old splitting criterion based on the
		//        size of the off-diagonal. A positive THRESH switches to splitting
		//        which preserves relative accuracy.
		if *tryrac {
			//           Test whether the matrix warrants the more expensive relative approach.
			Dlarrr(n, d, e, &iinfo)
		} else {
			//           The user does not care about relative accurately eigenvalues
			iinfo = -1
		}
		//        Set the splitting criterion
		if iinfo == 0 {
			thresh = eps
		} else {
			thresh = -eps
			//           relative accuracy is desired but T does not guarantee it
			(*tryrac) = false
		}

		if *tryrac {
			//           Copy original diagonal, needed to guarantee relative accuracy
			goblas.Dcopy(n, d, func() *int { y := 1; return &y }(), work.Off(indd-1), func() *int { y := 1; return &y }())
		}
		//        Store the squares of the offdiagonal values of T
		for j = 1; j <= (*n)-1; j++ {
			work.Set(inde2+j-1-1, math.Pow(e.Get(j-1), 2))
		}
		//        Set the tolerance parameters for bisection
		if !wantz {
			//           DLARRE computes the eigenvalues to full precision.
			rtol1 = four * eps
			rtol2 = four * eps
		} else {
			//           DLARRE computes the eigenvalues to less than full precision.
			//           ZLARRV will refine the eigenvalue approximations, and we only
			//           need less accurate initial bisection in DLARRE.
			//           Note: these settings do only affect the subset case and DLARRE
			rtol1 = math.Sqrt(eps)
			rtol2 = maxf64(math.Sqrt(eps)*5.0e-3, four*eps)
		}
		Dlarre(_range, n, &wl, &wu, &iil, &iiu, d, e, work.Off(inde2-1), &rtol1, &rtol2, &thresh, &nsplit, toSlice(iwork, iinspl-1), m, w, work.Off(inderr-1), work.Off(indgp-1), toSlice(iwork, iindbl-1), toSlice(iwork, iindw-1), work.Off(indgrs-1), &pivmin, work.Off(indwrk-1), toSlice(iwork, iindwk-1), &iinfo)
		if iinfo != 0 {
			(*info) = 10 + absint(iinfo)
			return
		}
		//        Note that if RANGE .NE. 'V', DLARRE computes bounds on the desired
		//        part of the spectrum. All desired eigenvalues are contained in
		//        (WL,WU]
		if wantz {
			//           Compute the desired eigenvectors corresponding to the computed
			//           eigenvalues
			Zlarrv(n, &wl, &wu, d, e, &pivmin, toSlice(iwork, iinspl-1), m, func() *int { y := 1; return &y }(), m, &minrgp, &rtol1, &rtol2, w, work.Off(inderr-1), work.Off(indgp-1), toSlice(iwork, iindbl-1), toSlice(iwork, iindw-1), work.Off(indgrs-1), z, ldz, isuppz, work.Off(indwrk-1), toSlice(iwork, iindwk-1), &iinfo)
			if iinfo != 0 {
				(*info) = 20 + absint(iinfo)
				return
			}
		} else {
			//           DLARRE computes eigenvalues of the (shifted) root representation
			//           ZLARRV returns the eigenvalues of the unshifted matrix.
			//           However, if the eigenvectors are not desired by the user, we need
			//           to apply the corresponding shifts from DLARRE to obtain the
			//           eigenvalues of the original matrix.
			for j = 1; j <= (*m); j++ {
				itmp = (*iwork)[iindbl+j-1-1]
				w.Set(j-1, w.Get(j-1)+e.Get((*iwork)[iinspl+itmp-1-1]-1))
			}
		}

		if *tryrac {
			//           Refine computed eigenvalues so that they are relatively accurate
			//           with respect to the original matrix T.
			ibegin = 1
			wbegin = 1
			for jblk = 1; jblk <= (*iwork)[iindbl+(*m)-1-1]; jblk++ {
				iend = (*iwork)[iinspl+jblk-1-1]
				in = iend - ibegin + 1
				wend = wbegin - 1
				//              check if any eigenvalues have to be refined in this block
			label36:
				;
				if wend < (*m) {
					if (*iwork)[iindbl+wend-1] == jblk {
						wend = wend + 1
						goto label36
					}
				}
				if wend < wbegin {
					ibegin = iend + 1
					goto label39
				}
				offset = (*iwork)[iindw+wbegin-1-1] - 1
				ifirst = (*iwork)[iindw+wbegin-1-1]
				ilast = (*iwork)[iindw+wend-1-1]
				rtol2 = four * eps
				Dlarrj(&in, work.Off(indd+ibegin-1-1), work.Off(inde2+ibegin-1-1), &ifirst, &ilast, &rtol2, &offset, w.Off(wbegin-1), work.Off(inderr+wbegin-1-1), work.Off(indwrk-1), toSlice(iwork, iindwk-1), &pivmin, &tnrm, &iinfo)
				ibegin = iend + 1
				wbegin = wend + 1
			label39:
			}
		}

		//        If matrix was scaled, then rescale eigenvalues appropriately.
		if scale != one {
			goblas.Dscal(m, toPtrf64(one/scale), w, func() *int { y := 1; return &y }())
		}
	}

	//     If eigenvalues are not in increasing order, then sort them,
	//     possibly along with eigenvectors.
	if nsplit > 1 || (*n) == 2 {
		if !wantz {
			Dlasrt('I', m, w, &iinfo)
			if iinfo != 0 {
				(*info) = 3
				return
			}
		} else {
			for j = 1; j <= (*m)-1; j++ {
				i = 0
				tmp = w.Get(j - 1)
				for jj = j + 1; jj <= (*m); jj++ {
					if w.Get(jj-1) < tmp {
						i = jj
						tmp = w.Get(jj - 1)
					}
				}
				if i != 0 {
					w.Set(i-1, w.Get(j-1))
					w.Set(j-1, tmp)
					if wantz {
						goblas.Zswap(n, z.CVector(0, i-1), func() *int { y := 1; return &y }(), z.CVector(0, j-1), func() *int { y := 1; return &y }())
						itmp = (*isuppz)[2*i-1-1]
						(*isuppz)[2*i-1-1] = (*isuppz)[2*j-1-1]
						(*isuppz)[2*j-1-1] = itmp
						itmp = (*isuppz)[2*i-1]
						(*isuppz)[2*i-1] = (*isuppz)[2*j-1]
						(*isuppz)[2*j-1] = itmp
					}
				}
			}
		}
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
