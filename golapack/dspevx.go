package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspevx computes selected eigenvalues and, optionally, eigenvectors
// of a real symmetric matrix A in packed storage.  Eigenvalues/vectors
// can be selected by specifying either a _range of values or a _range of
// indices for the desired eigenvalues.
func Dspevx(jobz, _range byte, uplo mat.MatUplo, n int, ap *mat.Vector, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.Matrix, work *mat.Vector, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, test, valeig, wantz bool
	var order byte
	var abstll, anrm, bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu, zero float64
	var i, imax, indd, inde, indee, indibl, indisp, indiwo, indtau, indwrk, iscale, itmp1, j, jj int

	zero = 0.0
	one = 1.0

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
				err = fmt.Errorf("n > 0 && vu <= vl: n=%v, vu=%v, vl=%v", n, vu, vl)
			}
		} else if indeig {
			if il < 1 || il > max(1, n) {
				err = fmt.Errorf("il < 1 || il > max(1, n): il=%v, n=%v", il, n)
			} else if iu < min(n, il) || iu > n {
				err = fmt.Errorf("iu < min(n, il) || iu > n: iu=%v, il=%v, n=%v", iu, il, n)
			}
		}
	}
	if err == nil {
		if z.Rows < 1 || (wantz && z.Rows < n) {
			err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dspevx", err)
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
			w.Set(0, ap.Get(0))
		} else {
			if vl < ap.Get(0) && vu >= ap.Get(0) {
				m = 1
				w.Set(0, ap.Get(0))
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
	} else {
		vll = zero
		vuu = zero
	}
	anrm = Dlansp('M', uplo, n, ap, work)
	if anrm > zero && anrm < rmin {
		iscale = 1
		sigma = rmin / anrm
	} else if anrm > rmax {
		iscale = 1
		sigma = rmax / anrm
	}
	if iscale == 1 {
		ap.Scal((n*(n+1))/2, sigma, 1)
		if abstol > 0 {
			abstll = abstol * sigma
		}
		if valeig {
			vll = vl * sigma
			vuu = vu * sigma
		}
	}

	//     Call DSPTRD to reduce symmetric packed matrix to tridiagonal form.
	indtau = 1
	inde = indtau + n
	indd = inde + n
	indwrk = indd + n
	if err = Dsptrd(uplo, n, ap, work.Off(indd-1), work.Off(inde-1), work.Off(indtau-1)); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or DOPGTR and SSTEQR.  If this fails
	//     for some eigenvalue, then try DSTEBZ.
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
			if err = Dopgtr(uplo, n, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
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
			goto label20
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
		if err = Dopmtr(Left, uplo, NoTrans, n, m, ap, work.Off(indtau-1), z, work.Off(indwrk-1)); err != nil {
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
				z.Off(0, j-1).Vector().Swap(n, z.Off(0, i-1).Vector(), 1, 1)
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
