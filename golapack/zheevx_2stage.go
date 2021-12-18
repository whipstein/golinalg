package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zheevx2stage computes selected eigenvalues and, optionally, eigenvectors
// of a complex Hermitian matrix A using the 2stage technique for
// the reduction to tridiagonal.  Eigenvalues and eigenvectors can
// be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
func Zheevx2stage(jobz, _range byte, uplo mat.MatUplo, n int, a *mat.CMatrix, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, lower, lquery, test, valeig, wantz bool
	var order byte
	var cone complex128
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, ib, imax, indd, inde, indee, indhous, indibl, indisp, indiwk, indrwk, indtau, indwrk, iscale, itmp1, j, jj, kd, lhtrd, llwork, lwmin, lwtrd int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

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
			work.SetRe(0, float64(lwmin))
		} else {
			kd = Ilaenv2stage(1, "Zhetrd2stage", []byte{jobz}, n, -1, -1, -1)
			ib = Ilaenv2stage(2, "Zhetrd2stage", []byte{jobz}, n, kd, -1, -1)
			lhtrd = Ilaenv2stage(3, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
			lwtrd = Ilaenv2stage(4, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
			lwmin = n + lhtrd + lwtrd
			work.SetRe(0, float64(lwmin))
		}

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zheevx2stage", err)
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
			w.Set(0, a.GetRe(0, 0))
		} else if valeig {
			if vl < a.GetRe(0, 0) && vu >= a.GetRe(0, 0) {
				m = 1
				w.Set(0, a.GetRe(0, 0))
			}
		}
		if wantz {
			z.Set(0, 0, cone)
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
	anrm = Zlanhe('M', uplo, n, a, rwork)
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

	//     Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form.
	indd = 1
	inde = indd + n
	indrwk = inde + n
	indtau = 1
	indhous = indtau + n
	indwrk = indhous + lhtrd
	llwork = lwork - indwrk + 1

	if err = Zhetrd2stage(jobz, uplo, n, a, rwork.Off(indd-1), rwork.Off(inde-1), work.Off(indtau-1), work.Off(indhous-1), lhtrd, work.Off(indwrk-1), llwork); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired and ABSTOL is less than or equal to
	//     zero, then call DSTERF or ZUNGTR and ZSTEQR.  If this fails for
	//     some eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if il == 1 && iu == n {
			test = true
		}
	}
	if (alleig || test) && (abstol <= zero) {
		w.Copy(n, rwork.Off(indd-1), 1, 1)
		indee = indrwk + 2*n
		if !wantz {
			rwork.Off(indee-1).Copy(n-1, rwork.Off(inde-1), 1, 1)
			if info, err = Dsterf(n, w, rwork.Off(indee-1)); err != nil {
				panic(err)
			}
		} else {
			Zlacpy(Full, n, n, a, z)
			if err = Zungtr(uplo, n, z, work.Off(indtau-1), work.Off(indwrk-1), llwork); err != nil {
				panic(err)
			}
			rwork.Off(indee-1).Copy(n-1, rwork.Off(inde-1), 1, 1)
			if info, err = Zsteqr(jobz, n, w, rwork.Off(indee-1), z, rwork.Off(indrwk-1)); err != nil {
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

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + n
	indiwk = indisp + n
	if m, _, info, err = Dstebz(_range, order, n, vll, vuu, il, iu, abstll, rwork.Off(indd-1), rwork.Off(inde-1), w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), rwork.Off(indrwk-1), toSlice(iwork, indiwk-1)); err != nil {
		panic(err)
	}

	if wantz {
		if info, err = Zstein(n, rwork.Off(indd-1), rwork.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, rwork.Off(indrwk-1), toSlice(iwork, indiwk-1), ifail); err != nil {
			panic(err)
		}

		//        Apply unitary matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by ZSTEIN.
		if err = Zunmtr(Left, uplo, NoTrans, n, m, a, work.Off(indtau-1), z, work.Off(indwrk-1), llwork); err != nil {
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
				z.Off(0, j-1).CVector().Swap(n, z.Off(0, i-1).CVector(), 1, 1)
				if info != 0 {
					itmp1 = (*ifail)[i-1]
					(*ifail)[i-1] = (*ifail)[j-1]
					(*ifail)[j-1] = itmp1
				}
			}
		}
	}

	//     Set WORK(1) to optimal complex workspace size.
	work.SetRe(0, float64(lwmin))

	return
}
