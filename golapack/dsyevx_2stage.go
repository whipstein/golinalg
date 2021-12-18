package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsyevx2stage computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A using the 2stage technique for
// the reduction to tridiagonal.  Eigenvalues and eigenvectors can be
// selected by specifying either a _range of values or a _range of indices
// for the desired eigenvalues.
func Dsyevx2stage(jobz, _range byte, uplo mat.MatUplo, n int, a *mat.Matrix, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, lower, lquery, test, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, ib, imax, indd, inde, indee, indhous, indibl, indisp, indiwo, indtau, indwkn, indwrk, iscale, itmp1, j, jj, kd, lhtrd, llwork, llwrkn, lwmin, lwtrd int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	lower = uplo == Lower
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lquery = (lwork == -1)

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
		if n <= 1 {
			lwmin = 1
			work.Set(0, float64(lwmin))
		} else {
			kd = Ilaenv2stage(1, "Dsytrd2stage", []byte{jobz}, n, -1, -1, -1)
			ib = Ilaenv2stage(2, "Dsytrd2stage", []byte{jobz}, n, kd, -1, -1)
			lhtrd = Ilaenv2stage(3, "Dsytrd2stage", []byte{jobz}, n, kd, ib, -1)
			lwtrd = Ilaenv2stage(4, "Dsytrd2stage", []byte{jobz}, n, kd, ib, -1)
			lwmin = max(8*n, 3*n+lhtrd+lwtrd)
			work.Set(0, float64(lwmin))
		}

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsyevx2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	m = 0
	if n == 0 {
		return
	}

	if n == 1 {
		if alleig || indeig {
			m = 1
			w.Set(0, a.Get(0, 0))
		} else {
			if vl < a.Get(0, 0) && vu >= a.Get(0, 0) {
				m = 1
				w.Set(0, a.Get(0, 0))
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
	rmax = math.Min(math.Sqrt(bignum), one/math.Sqrt(math.Sqrt(safmin)))

	//     Scale matrix to allowable _range, if necessary.
	iscale = 0
	abstll = abstol
	if valeig {
		vll = vl
		vuu = vu
	}
	anrm = Dlansy('M', uplo, n, a, work)
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
				a.Off(j-1, j-1).Vector().Scal(n-j+1, sigma, 1)
			}
		} else {
			for j = 1; j <= n; j++ {
				a.Off(0, j-1).Vector().Scal(j, sigma, 1)
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

	//     Call DSYTRD_2STAGE to reduce symmetric matrix to tridiagonal form.
	indtau = 1
	inde = indtau + n
	indd = inde + n
	indhous = indd + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1

	if err = Dsytrd2stage(jobz, uplo, n, a, work.Off(indd-1), work.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired and ABSTOL is less than or equal to
	//     zero, then call DSTERF or DORGTR and SSTEQR.  If this fails for
	//     some eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if il == 1 && iu == n {
			test = true
		}
	}
	if (alleig || test) && (abstol <= zero) {
		w.Copy(n, work.Off(indd-1), 1, 1)
		indee = indwrk + 2*n
		if !wantz {
			work.Off(indee-1).Copy(n-1, work.Off(inde-1), 1, 1)
			if info, err = Dsterf(n, w, work.Off(indee-1)); err != nil {
				panic(err)
			}
		} else {
			Dlacpy(Full, n, n, a, z)
			if err = Dorgtr(uplo, n, z, work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
				panic(err)
			}
			work.Off(indee-1).Copy(n-1, work.Off(inde-1), 1, 1)
			if info, err = Dsteqr(jobz, n, w, work.Off(indee-1), z, work.Off(indwrk-1)); err != nil {
				panic(err)
			}
			if info == 0 {
				for i = 1; i <= n; i++ {
					(*ifail)[i-1] = 0
				}
			}
		}
		if info == 0 {
			m = n
			goto label40
		}
		info = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, SSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + n
	indiwo = indisp + n
	if m, _, info, err = Dstebz(_range, order, n, vll, vuu, il, iu, abstll, work.Off(indd-1), work.Off(inde-1), w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwrk-1), toSlice(iwork, indiwo-1)); err != nil {
		panic(err)
	}

	if wantz {
		if info, err = Dstein(n, work.Off(indd-1), work.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, work.Off(indwrk-1), toSlice(iwork, indiwo-1), ifail); err != nil {
			panic(err)
		}

		//        Apply orthogonal matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by DSTEIN.
		indwkn = inde
		llwrkn = lwork - indwkn + 1
		if err = Dormtr(Left, uplo, NoTrans, n, m, a, work.Off(indtau-1), z, work.Off(indwkn-1), llwrkn); err != nil {
			panic(err)
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label40:
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
				z.Off(0, j-1).Vector().Swap(n, z.Off(0, i-1).Vector(), 1, 1)
				if info != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwmin))

	return
}
