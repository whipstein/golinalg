package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpevx computes selected eigenvalues and, optionally, eigenvectors
// of a complex Hermitian matrix A in packed storage.
// Eigenvalues/vectors can be selected by specifying either a _range of
// values or a _range of indices for the desired eigenvalues.
func Zhpevx(jobz, _range byte, uplo mat.MatUplo, n int, ap *mat.CVector, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, rwork *mat.Vector, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, test, valeig, wantz bool
	var order byte
	var cone complex128
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, imax, indd, inde, indee, indibl, indisp, indiwk, indrwk, indtau, indwrk, iscale, itmp1, j, jj int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(alleig || valeig || indeig) {
		err = fmt.Errorf("!(alleig || valeig || indeig): _range='%c'", _range)
	} else if !(uplo == Lower || uplo == Upper) {
		err = fmt.Errorf("!(uplo == Lower || uplo == Upper): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
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

	if err != nil {
		gltest.Xerbla2("Zhpevx", err)
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
			w.Set(0, ap.GetRe(0))
		} else {
			if vl < ap.GetRe(0) && vu >= ap.GetRe(0) {
				m = 1
				w.Set(0, ap.GetRe(0))
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
	} else {
		vll = zero
		vuu = zero
	}
	anrm = Zlanhp('M', uplo, n, ap, rwork)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		ap.Dscal((n*(n+1))/2, sigma, 1)
		if abstol > 0 {
			abstll = abstol * sigma
		}
		if valeig {
			vll = vl * sigma
			vuu = vu * sigma
		}
	}

	//     Call ZHPTRD to reduce Hermitian packed matrix to tridiagonal form.
	indd = 1
	inde = indd + n
	indrwk = inde + n
	indtau = 1
	indwrk = indtau + n
	if err = Zhptrd(uplo, n, ap, rwork.Off(indd-1), rwork.Off(inde-1), work.Off(indtau-1)); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or ZUPGTR and ZSTEQR.  If this fails
	//     for some eigenvalue, then try DSTEBZ.
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
			if err = Zupgtr(uplo, n, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
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
			goto label20
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
		indwrk = indtau + n
		if err = Zupmtr(Left, uplo, NoTrans, n, m, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
			panic(err)
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
label20:
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

	return
}
