package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheevr2stage computes selected eigenvalues and, optionally, eigenvectors
// of a complex Hermitian matrix A using the 2stage technique for
// the reduction to tridiagonal.  Eigenvalues and eigenvectors can
// be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
//
// Zheevr2stage first reduces the matrix A to tridiagonal form T with a call
// to ZHETRD.  Then, whenever possible, Zheevr2stage calls ZSTEMR to compute
// eigenspectrum using Relatively Robust Representations.  ZSTEMR
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
// Note 1 : Zheevr2stage calls ZSTEMR when the full spectrum is requested
// on machines which conform to the ieee-754 floating point standard.
// Zheevr2stage calls DSTEBZ and ZSTEIN on non-ieee machines and
// when partial spectrum requests are made.
//
// Normal execution of ZSTEMR may create NaNs and infinities and
// hence may abort due to a floating point exception in environments
// which do not handle NaNs and infinities in the ieee standard default
// manner.
func Zheevr2stage(jobz, _range byte, uplo mat.MatUplo, n int, a *mat.CMatrix, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.CMatrix, isuppz *[]int, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (m, info int, err error) {
	var alleig, indeig, lower, lquery, test, tryrac, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, two, vll, vuu, zero float64
	var i, ib, ieeeok, imax, indhous, indibl, indifl, indisp, indiwo, indrd, indrdd, indre, indree, indrwk, indtau, indwk, indwkn, iscale, itmp1, j, jj, kd, lhtrd, liwmin, llrwork, llwork, llwrkn, lrwmin, lwmin, lwtrd int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	ieeeok = Ilaenv(10, "Zheevr", []byte{'N'}, 1, 2, 3, 4)

	lower = uplo == Lower
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	lquery = ((lwork == -1) || (lrwork == -1) || (liwork == -1))

	kd = Ilaenv2stage(1, "Zhetrd2stage", []byte{jobz}, n, -1, -1, -1)
	ib = Ilaenv2stage(2, "Zhetrd2stage", []byte{jobz}, n, kd, -1, -1)
	lhtrd = Ilaenv2stage(3, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
	lwtrd = Ilaenv2stage(4, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
	lwmin = n + lhtrd + lwtrd
	lrwmin = max(1, 24*n)
	liwmin = max(1, 10*n)

	if jobz != 'N' {
		err = fmt.Errorf("jobz != 'N': jobz='%c'", jobz)
	} else if !(alleig || valeig || indeig) {
		err = fmt.Errorf("!(alleig || valeig || indeig): _range='%c'", _range)
	} else if !(lower || uplo == Upper) {
		err = fmt.Errorf("!(lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else {
		if valeig {
			if n > 0 && vu <= vl {
				err = fmt.Errorf("n > 0 && vu <= vl: n=%v, vl=%v, vu=%v", n, vl, vu)
			}
		} else if indeig {
			if il < 1 || il > max(1, n) {
				err = fmt.Errorf("il < 1 || il > max(1, n): n=%v, il=%v", n, il)
			} else if iu < min(n, il) || iu > n {
				err = fmt.Errorf("iu < min(n, il) || iu > n: n=%v, il=%v, iu=%v", n, il, iu)
			}
		}
	}
	if err == nil {
		if z.Rows < 1 || (wantz && z.Rows < n) {
			err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
		}
	}

	if err == nil {
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if lrwork < lrwmin && !lquery {
			err = fmt.Errorf("lrwork < lrwmin && !lquery: lrwork=%v, lrwmin=%v, lquery=%v", lrwork, lrwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zheevr2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	m = 0
	if n == 0 {
		work.Set(0, 1)
		return
	}

	if n == 1 {
		work.Set(0, 2)
		if alleig || indeig {
			m = 1
			w.Set(0, a.GetRe(0, 0))
		} else {
			if vl < a.GetRe(0, 0) && vu >= a.GetRe(0, 0) {
				m = 1
				w.Set(0, a.GetRe(0, 0))
			}
		}
		if wantz {
			z.SetRe(0, 0, one)
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
	abstll = abstol
	if valeig {
		vll = vl
		vuu = vu
	}
	anrm = Zlansy('M', uplo, n, a, rwork)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		if lower {
			for j = 1; j <= n; j++ {
				a.Off(j-1, j-1).CVector().Dscal(n-j+1, sigma, 1)
			}
		} else {
			for j = 1; j <= n; j++ {
				a.Off(0, j-1).CVector().Dscal(j, sigma, 1)
			}
		}
		if abstol > 0 {
			abstll = abstol * sigma
		}
		if valeig {
			vll = vl * sigma
			vuu = vu * sigma
		}
	}
	//     Initialize indices into workspaces.  Note: The IWORK indices are
	//     used only if DSTERF or ZSTEMR fail.
	//     WORK(INDTAU:INDTAU+N-1) stores the complex scalar factors of the
	//     elementary reflectors used in ZHETRD.
	indtau = 1
	//     INDWK is the starting offset of the remaining complex workspace,
	//     and LLWORK is the remaining complex workspace size.
	indhous = indtau + n
	indwk = indhous + lhtrd
	llwork = lwork - indwk + 1
	//     RWORK(INDRD:INDRD+N-1) stores the real tridiagonal's diagonal
	//     entries.
	indrd = 1
	//     RWORK(INDRE:INDRE+N-1) stores the off-diagonal entries of the
	//     tridiagonal matrix from ZHETRD.
	indre = indrd + n
	//     RWORK(INDRDD:INDRDD+N-1) is a copy of the diagonal entries over
	//     -written by ZSTEMR (the DSTERF path copies the diagonal to W).
	indrdd = indre + n
	//     RWORK(INDREE:INDREE+N-1) is a copy of the off-diagonal entries over
	//     -written while computing the eigenvalues in DSTERF and ZSTEMR.
	indree = indrdd + n
	//     INDRWK is the starting offset of the left-over real workspace, and
	//     LLRWORK is the remaining workspace size.
	indrwk = indree + n
	llrwork = lrwork - indrwk + 1
	//     IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
	//     stores the block indices of each of the M<=N eigenvalues.
	indibl = 1
	//     IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
	//     stores the starting and finishing indices of each block.
	indisp = indibl + n
	//     IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
	//     that corresponding to eigenvectors that fail to converge in
	//     ZSTEIN.  This information is discarded; if any fail, the driver
	//     returns INFO > 0.
	indifl = indisp + n
	//     INDIWO is the offset of the remaining integer workspace.
	indiwo = indifl + n

	//     Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form.
	if err = Zhetrd2stage(jobz, uplo, n, a, rwork.Off(indrd-1), rwork.Off(indre-1), work.Off(indtau-1), work.Off(indhous-1), lhtrd, work.Off(indwk-1), llwork); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired
	//     then call DSTERF or ZSTEMR and ZUNMTR.
	test = false
	if indeig {
		if il == 1 && iu == n {
			test = true
		}
	}
	if (alleig || test) && (ieeeok == 1) {
		if !wantz {
			w.Copy(n, rwork.Off(indrd-1), 1, 1)
			rwork.Off(indree-1).Copy(n-1, rwork.Off(indre-1), 1, 1)
			if info, err = Dsterf(n, w, rwork.Off(indree-1)); err != nil {
				panic(err)
			}
		} else {
			rwork.Off(indree-1).Copy(n-1, rwork.Off(indre-1), 1, 1)
			rwork.Off(indrdd-1).Copy(n, rwork.Off(indrd-1), 1, 1)

			if abstol <= two*float64(n)*eps {
				tryrac = true
			} else {
				tryrac = false
			}
			if m, tryrac, info, err = Zstemr(jobz, 'A', n, rwork.Off(indrdd-1), rwork.Off(indree-1), vl, vu, il, iu, w, z, n, isuppz, tryrac, rwork.Off(indrwk-1), llrwork, iwork, liwork); err != nil {
				panic(err)
			}

			//           Apply unitary matrix used in reduction to tridiagonal
			//           form to eigenvectors returned by ZSTEMR.
			if wantz && info == 0 {
				indwkn = indwk
				llwrkn = lwork - indwkn + 1
				if err = Zunmtr(Left, uplo, NoTrans, n, m, a, work.Off(indtau-1), z, work.Off(indwkn-1), llwrkn); err != nil {
					panic(err)
				}
			}
		}

		if info == 0 {
			m = n
			goto label30
		}
		info = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN.
	//     Also call DSTEBZ and ZSTEIN if ZSTEMR fails.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	if m, _, info, err = Dstebz(_range, order, n, vll, vuu, il, iu, abstll, rwork.Off(indrd-1), rwork.Off(indre-1), w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), rwork.Off(indrwk-1), toSlice(iwork, indiwo-1)); err != nil {
		panic(err)
	}

	if wantz {
		if info, err = Zstein(n, rwork.Off(indrd-1), rwork.Off(indre-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, rwork.Off(indrwk-1), toSlice(iwork, indiwo-1), toSlice(iwork, indifl-1)); err != nil {
			panic(err)
		}

		//        Apply unitary matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by ZSTEIN.
		indwkn = indwk
		llwrkn = lwork - indwkn + 1
		if err = Zunmtr(Left, uplo, NoTrans, n, m, a, work.Off(indtau-1), z, work.Off(indwkn-1), llwrkn); err != nil {
			panic(err)
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label30:
	;
	if iscale == 1 {
		if info == 0 {
			imax = m
		} else {
			imax = info - 1
		}
		w.Scal(imax, one/sigma, 1)
	}

	//     If eigenvalues are not in order, then sort them, along with
	//     eigenvectors.
	if wantz {
		for j = 1; j <= m-1; j++ {
			i = 0
			tmp1 = w.Get(j - 1)
			for jj = j + 1; jj <= m; jj++ {
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
				z.Off(0, j-1).CVector().Swap(n, z.Off(0, i-1).CVector(), 1, 1)
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin

	return
}
