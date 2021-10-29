package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbgvx computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite banded eigenproblem, of
// the form A*x=(lambda)*B*x.  Here A and B are assumed to be symmetric
// and banded, and B is also positive definite.  Eigenvalues and
// eigenvectors can be selected by specifying either all eigenvalues,
// a _range of values or a _range of indices for the desired eigenvalues.
func Dsbgvx(jobz, _range byte, uplo mat.MatUplo, n, ka, kb int, ab, bb, q *mat.Matrix, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.Matrix, work *mat.Vector, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, test, upper, valeig, wantz bool
	var order, vect byte
	var one, tmp1, zero float64
	var i, indd, inde, indee, indibl, indisp, indiwo, indwrk, itmp1, j, jj int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(alleig || valeig || indeig) {
		err = fmt.Errorf("!(alleig || valeig || indeig): _range='%c'", _range)
	} else if !(upper || uplo == Lower) {
		err = fmt.Errorf("!(upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ka < 0 {
		err = fmt.Errorf("ka < 0: ka=%v", ka)
	} else if kb < 0 || kb > ka {
		err = fmt.Errorf("kb < 0 || kb > ka: kb=%v, ka=%v", kb, ka)
	} else if ab.Rows < ka+1 {
		err = fmt.Errorf("ab.Rows < ka+1: ab.Rows=%v, ka=%v", ab.Rows, ka)
	} else if bb.Rows < kb+1 {
		err = fmt.Errorf("bb.Rows < kb+1: bb.Rows=%v, kb=%v", bb.Rows, kb)
	} else if q.Rows < 1 || (wantz && q.Rows < n) {
		err = fmt.Errorf("q.Rows < 1 || (wantz && q.Rows < n): jobz='%c', q.Rows=%v, n=%v", jobz, q.Rows, n)
	} else {
		if valeig {
			if n > 0 && vu <= vl {
				err = fmt.Errorf("n > 0 && vu <= vl: n=%v, vu=%v, vl=%v", n, vu, vl)
			}
		} else if indeig {
			if il < 1 || il > max(1, n) {
				err = fmt.Errorf("il < 1 || il > max(1, n): il=%v, iu=%v, n=%v", il, iu, n)
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
		gltest.Xerbla2("Dsbgvx", err)
		return
	}

	//     Quick return if possible
	m = 0
	if n == 0 {
		return
	}

	//     Form a split Cholesky factorization of B.
	if info, err = Dpbstf(uplo, n, kb, bb); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem.
	if err = Dsbgst(jobz, uplo, n, ka, kb, ab, bb, q, work); err != nil {
		panic(err)
	}

	//     Reduce symmetric band matrix to tridiagonal form.
	indd = 1
	inde = indd + n
	indwrk = inde + n
	if wantz {
		vect = 'U'
	} else {
		vect = 'N'
	}
	if err = Dsbtrd(vect, uplo, n, ka, ab, work.Off(indd-1), work.Off(inde-1), q, work.Off(indwrk-1)); err != nil {
		panic(err)
	}

	//     If all eigenvalues are desired and ABSTOL is less than or equal
	//     to zero, then call DSTERF or SSTEQR.  If this fails for some
	//     eigenvalue, then try DSTEBZ.
	test = false
	if indeig {
		if il == 1 && iu == n {
			test = true
		}
	}
	if (alleig || test) && (abstol <= zero) {
		goblas.Dcopy(n, work.Off(indd-1, 1), w.Off(0, 1))
		indee = indwrk + 2*n
		goblas.Dcopy(n-1, work.Off(inde-1, 1), work.Off(indee-1, 1))
		if !wantz {
			if info, err = Dsterf(n, w, work.Off(indee-1)); err != nil {
				panic(err)
			}
		} else {
			Dlacpy(Full, n, n, q, z)
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
			goto label30
		}
		info = 0
	}

	//     Otherwise, call DSTEBZ and, if eigenvectors are desired,
	//     call DSTEIN.
	if wantz {
		order = 'B'
	} else {
		order = 'E'
	}
	indibl = 1
	indisp = indibl + n
	indiwo = indisp + n
	if m, _, info, err = Dstebz(_range, order, n, vl, vu, il, iu, abstol, work.Off(indd-1), work.Off(inde-1), w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), work.Off(indwrk-1), toSlice(iwork, indiwo-1)); err != nil {
		panic(err)
	}

	if wantz {
		if info, err = Dstein(n, work.Off(indd-1), work.Off(inde-1), m, w, toSlice(iwork, indibl-1), toSlice(iwork, indisp-1), z, work.Off(indwrk-1), toSlice(iwork, indiwo-1), ifail); err != nil {
			panic(err)
		}

		//        Apply transformation matrix used in reduction to tridiagonal
		//        form to eigenvectors returned by DSTEIN.
		for j = 1; j <= m; j++ {
			goblas.Dcopy(n, z.Vector(0, j-1, 1), work.Off(0, 1))
			if err = goblas.Dgemv(NoTrans, n, n, one, q, work.Off(0, 1), zero, z.Vector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}
	}

label30:
	;

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
				goblas.Dswap(n, z.Vector(0, i-1, 1), z.Vector(0, j-1, 1))
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
