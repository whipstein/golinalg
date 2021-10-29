package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygvx computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
// and B are assumed to be symmetric and B is also positive definite.
// Eigenvalues and eigenvectors can be selected by specifying either a
// _range of values or a _range of indices for the desired eigenvalues.
func Dsygvx(itype int, jobz, _range byte, uplo mat.MatUplo, n int, a, b *mat.Matrix, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, lquery, upper, valeig, wantz bool
	var trans mat.MatTrans
	var one float64
	var lwkmin, lwkopt, nb int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'
	lquery = (lwork == -1)

	if itype < 1 || itype > 3 {
		err = fmt.Errorf("itype < 1 || itype > 3: itype=%v", itype)
	} else if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(alleig || valeig || indeig) {
		err = fmt.Errorf("!(alleig || valeig || indeig): _range='%c'", _range)
	} else if !(upper || uplo == Lower) {
		err = fmt.Errorf("!(upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
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
		lwkmin = max(1, 8*n)
		nb = Ilaenv(1, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = max(lwkmin, (nb+3)*n)
		work.Set(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsygvx", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	m = 0
	if n == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	if info, err = Dpotrf(uplo, n, b); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	if err = Dsygst(itype, uplo, n, a, b); err != nil {
		panic(err)
	}
	if m, info, err = Dsyevx(jobz, _range, uplo, n, a, vl, vu, il, iu, abstol, w, z, work, lwork, iwork, ifail); err != nil {
		panic(err)
	}

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		if info > 0 {
			m = info - 1
		}
		if itype == 1 || itype == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = NoTrans
			} else {
				trans = Trans
			}

			if err = goblas.Dtrsm(Left, uplo, trans, NonUnit, n, m, one, b, z); err != nil {
				panic(err)
			}

		} else if itype == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = Trans
			} else {
				trans = NoTrans
			}

			if err = goblas.Dtrmm(Left, uplo, trans, NonUnit, n, m, one, b, z); err != nil {
				panic(err)
			}
		}
	}

	//     Set WORK(1) to optimal workspace size.
	work.Set(0, float64(lwkopt))

	return
}
