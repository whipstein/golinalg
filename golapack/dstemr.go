package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstemr computes selected eigenvalues and, optionally, eigenvectors
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
// 1.Dstemr works only on machines which follow IEEE-754
// floating-point standard in their handling of infinities and NaNs.
// This permits the use of efficient inner loops avoiding a check for
// zero divisors.
func Dstemr(jobz, _range byte, n int, d, e *mat.Vector, vl, vu float64, il, iu int, w *mat.Vector, z *mat.Matrix, nzc int, isuppz *[]int, tryrac bool, work *mat.Vector, lwork int, iwork *[]int, liwork int) (m int, tryracOut bool, info int, err error) {
	var alleig, indeig, lquery, valeig, wantz, zquery bool
	var bignum, cs, eps, four, minrgp, one, pivmin, r1, r2, rmax, rmin, rtol1, rtol2, safmin, scale, smlnum, sn, thresh, tmp, tnrm, wl, wu, zero float64
	var i, ibegin, iend, ifirst, iil, iindbl, iindw, iindwk, iinfo, iinspl, iiu, ilast, in, indd, inde2, inderr, indgp, indgrs, indwrk, itmp, j, jblk, jj, liwmin, lwmin, nsplit, nzcmin, offset, wbegin, wend int

	zero = 0.0
	one = 1.0
	four = 4.0
	minrgp = 1.0e-3
	tryracOut = tryrac

	//     Test the input parameters.
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	lquery = ((lwork == -1) || (liwork == -1))
	zquery = (nzc == -1)
	//     Dstemr needs WORK of size 6*N, IWORK of size 3*N.
	//     In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N.
	//     Furthermore, DLARRV needs WORK of size 12*N, IWORK of size 7*N.
	if wantz {
		lwmin = 18 * n
		liwmin = 10 * n
	} else {
		//        need less workspace if only the eigenvalues are wanted
		lwmin = 12 * n
		liwmin = 8 * n
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
		wl = vl
		wu = vu
	} else if indeig {
		//        We do not reference IL, IU in the cases RANGE = 'V','A'
		iil = il
		iiu = iu
	}

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(alleig || valeig || indeig) {
		err = fmt.Errorf("!(alleig || valeig || indeig): _range='%c'", _range)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if valeig && n > 0 && wu <= wl {
		err = fmt.Errorf("valeig && n > 0 && wu <= wl: _range='%c', n=%v, wl=%v, wu=%v", _range, n, wl, wu)
	} else if indeig && (iil < 1 || iil > n) {
		err = fmt.Errorf("indeig && (iil < 1 || iil > n): _range='%c', iil=%v, n=%v", _range, iil, n)
	} else if indeig && (iiu < iil || iiu > n) {
		err = fmt.Errorf("indeig && (iiu < iil || iiu > n): _range='%c', iil=%v, iiu=%v, n=%v", _range, iil, iiu, n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	} else if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	} else if liwork < liwmin && !lquery {
		err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = math.Min(math.Sqrt(bignum), one/math.Sqrt(math.Sqrt(safmin)))

	if err == nil {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if wantz && alleig {
			nzcmin = n
		} else if wantz && valeig {
			nzcmin, itmp, _ = Dlarrc('T', n, vl, vu, d, e, safmin)
		} else if wantz && indeig {
			nzcmin = iiu - iil + 1
		} else {
			//           WANTZ .EQ. FALSE.
			nzcmin = 0
		}
		if zquery && err == nil {
			z.Set(0, 0, float64(nzcmin))
		} else if nzc < nzcmin && !zquery {
			err = fmt.Errorf("nzc < nzcmin && !zquery: nzc=%v, nzcmin=%v, zquery=%v", nzc, nzcmin, zquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dstemr", err)
		return
	} else if lquery || zquery {
		return
	}

	//     Handle N = 0, 1, and 2 cases immediately
	m = 0
	if n == 0 {
		return
	}

	if n == 1 {
		if alleig || indeig {
			m = 1
			w.Set(0, d.Get(0))
		} else {
			if wl < d.Get(0) && wu >= d.Get(0) {
				m = 1
				w.Set(0, d.Get(0))
			}
		}
		if wantz && (!zquery) {
			z.Set(0, 0, one)
			(*isuppz)[0] = 1
			(*isuppz)[1] = 1
		}
		return
	}

	if n == 2 {
		if !wantz {
			r1, r2 = Dlae2(d.Get(0), e.Get(0), d.Get(1))
		} else if wantz && (!zquery) {
			r1, r2, cs, sn = Dlaev2(d.Get(0), e.Get(0), d.Get(1))
		}
		if alleig || (valeig && (r2 > wl) && (r2 <= wu)) || (indeig && (iil == 1)) {
			m = m + 1
			w.Set(m-1, r2)
			if wantz && (!zquery) {
				z.Set(0, m-1, -sn)
				z.Set(1, m-1, cs)
				//              Note: At most one of SN and CS can be zero.
				if sn != zero {
					if cs != zero {
						(*isuppz)[2*m-1-1] = 1
						(*isuppz)[2*m-1] = 2
					} else {
						(*isuppz)[2*m-1-1] = 1
						(*isuppz)[2*m-1] = 1
					}
				} else {
					(*isuppz)[2*m-1-1] = 2
					(*isuppz)[2*m-1] = 2
				}
			}
		}
		if alleig || (valeig && (r1 > wl) && (r1 <= wu)) || (indeig && (iiu == 2)) {
			m = m + 1
			w.Set(m-1, r1)
			if wantz && (!zquery) {
				z.Set(0, m-1, cs)
				z.Set(1, m-1, sn)
				//              Note: At most one of SN and CS can be zero.
				if sn != zero {
					if cs != zero {
						(*isuppz)[2*m-1-1] = 1
						(*isuppz)[2*m-1] = 2
					} else {
						(*isuppz)[2*m-1-1] = 1
						(*isuppz)[2*m-1] = 1
					}
				} else {
					(*isuppz)[2*m-1-1] = 2
					(*isuppz)[2*m-1] = 2
				}
			}
		}
	} else {
		//     Continue with general N
		indgrs = 1
		inderr = 2*n + 1
		indgp = 3*n + 1
		indd = 4*n + 1
		inde2 = 5*n + 1
		indwrk = 6*n + 1

		iinspl = 1
		iindbl = n + 1
		iindw = 2*n + 1
		iindwk = 3*n + 1

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
			goblas.Dscal(n, scale, d.Off(0, 1))
			goblas.Dscal(n-1, scale, e.Off(0, 1))
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
		if tryracOut {
			//           Test whether the matrix warrants the more expensive relative approach.
			iinfo = Dlarrr(n, d, e)
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
			tryracOut = false
		}

		if tryracOut {
			//           Copy original diagonal, needed to guarantee relative accuracy
			goblas.Dcopy(n, d.Off(0, 1), work.Off(indd-1, 1))
		}
		//        Store the squares of the offdiagonal values of T
		for j = 1; j <= n-1; j++ {
			work.Set(inde2+j-1-1, math.Pow(e.Get(j-1), 2))
		}
		//        Set the tolerance parameters for bisection
		if !wantz {
			//           DLARRE computes the eigenvalues to full precision.
			rtol1 = four * eps
			rtol2 = four * eps
		} else {
			//           DLARRE computes the eigenvalues to less than full precision.
			//           DLARRV will refine the eigenvalue approximations, and we can
			//           need less accurate initial bisection in DLARRE.
			//           Note: these settings do only affect the subset case and DLARRE
			rtol1 = math.Sqrt(eps)
			rtol2 = math.Max(math.Sqrt(eps)*5.0e-3, four*eps)
		}
		if wl, wu, nsplit, m, pivmin, iinfo, err = Dlarre(_range, n, wl, wu, iil, iiu, d, e, work.Off(inde2-1), rtol1, rtol2, thresh, toSlice(iwork, iinspl-1), w, work.Off(inderr-1), work.Off(indgp-1), toSlice(iwork, iindbl-1), toSlice(iwork, iindw-1), work.Off(indgrs-1), work.Off(indwrk-1), toSlice(iwork, iindwk-1)); err != nil {
			panic(err)
		}
		if iinfo != 0 {
			info = 10 + abs(iinfo)
			return
		}
		//        Note that if RANGE .NE. 'V', DLARRE computes bounds on the desired
		//        part of the spectrum. All desired eigenvalues are contained in
		//        (WL,WU]
		if wantz {
			//           Compute the desired eigenvectors corresponding to the computed
			//           eigenvalues
			iinfo = Dlarrv(n, wl, wu, d, e, pivmin, toSlice(iwork, iinspl-1), m, 1, m, minrgp, rtol1, rtol2, w, work.Off(inderr-1), work.Off(indgp-1), toSlice(iwork, iindbl-1), toSlice(iwork, iindw-1), work.Off(indgrs-1), z, isuppz, work.Off(indwrk-1), toSlice(iwork, iindwk-1))
			if iinfo != 0 {
				info = 20 + abs(iinfo)
				return
			}
		} else {
			//           DLARRE computes eigenvalues of the (shifted) root representation
			//           DLARRV returns the eigenvalues of the unshifted matrix.
			//           However, if the eigenvectors are not desired by the user, we need
			//           to apply the corresponding shifts from DLARRE to obtain the
			//           eigenvalues of the original matrix.
			for j = 1; j <= m; j++ {
				itmp = (*iwork)[iindbl+j-1-1]
				w.Set(j-1, w.Get(j-1)+e.Get((*iwork)[iinspl+itmp-1-1]-1))
			}
		}

		if tryracOut {
			//           Refine computed eigenvalues so that they are relatively accurate
			//           with respect to the original matrix T.
			ibegin = 1
			wbegin = 1
			for jblk = 1; jblk <= (*iwork)[iindbl+m-1-1]; jblk++ {
				iend = (*iwork)[iinspl+jblk-1-1]
				in = iend - ibegin + 1
				wend = wbegin - 1
				//              check if any eigenvalues have to be refined in this block
			label36:
				;
				if wend < m {
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
				Dlarrj(in, work.Off(indd+ibegin-1-1), work.Off(inde2+ibegin-1-1), ifirst, ilast, rtol2, offset, w.Off(wbegin-1), work.Off(inderr+wbegin-1-1), work.Off(indwrk-1), toSlice(iwork, iindwk-1), pivmin, tnrm)
				ibegin = iend + 1
				wbegin = wend + 1
			label39:
			}
		}

		//        If matrix was scaled, then rescale eigenvalues appropriately.
		if scale != one {
			goblas.Dscal(m, one/scale, w.Off(0, 1))
		}
	}

	//     If eigenvalues are not in increasing order, then sort them,
	//     possibly along with eigenvectors.
	if nsplit > 1 || n == 2 {
		if !wantz {
			if err = Dlasrt('I', m, w); err != nil {
				info = 3
				return
			}
		} else {
			for j = 1; j <= m-1; j++ {
				i = 0
				tmp = w.Get(j - 1)
				for jj = j + 1; jj <= m; jj++ {
					if w.Get(jj-1) < tmp {
						i = jj
						tmp = w.Get(jj - 1)
					}
				}
				if i != 0 {
					w.Set(i-1, w.Get(j-1))
					w.Set(j-1, tmp)
					if wantz {
						goblas.Dswap(n, z.Vector(0, i-1, 1), z.Vector(0, j-1, 1))
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

	return
}
