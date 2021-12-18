package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspgvx computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
// and B are assumed to be symmetric, stored in packed storage, and B
// is also positive definite.  Eigenvalues and eigenvectors can be
// selected by specifying either a _range of values or a _range of indices
// for the desired eigenvalues.
func Dspgvx(itype int, jobz, _range byte, uplo mat.MatUplo, n int, ap, bp *mat.Vector, vl, vu float64, il, iu int, abstol float64, w *mat.Vector, z *mat.Matrix, work *mat.Vector, iwork, ifail *[]int) (m, info int, err error) {
	var alleig, indeig, upper, valeig, wantz bool
	var trans mat.MatTrans
	var j int

	//     Test the input parameters.
	upper = uplo == Upper
	wantz = jobz == 'V'
	alleig = _range == 'A'
	valeig = _range == 'V'
	indeig = _range == 'I'

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
	} else {
		if valeig {
			if n > 0 && vu <= vl {
				err = fmt.Errorf("n > 0 && vu <= vl: n=%v, vu=%v, vl=%v", n, vu, vl)
			}
		} else if indeig {
			if il < 1 {
				err = fmt.Errorf("il < 1: il=%v", il)
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
		gltest.Xerbla2("Dspgvx", err)
		return
	}

	//     Quick return if possible
	m = 0
	if n == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	if info, err = Dpptrf(uplo, n, bp); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	if err = Dspgst(itype, uplo, n, ap, bp); err != nil {
		panic(err)
	}
	if m, info, err = Dspevx(jobz, _range, uplo, n, ap, vl, vu, il, iu, abstol, w, z, work, iwork, ifail); err != nil {
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

			for j = 1; j <= m; j++ {
				if err = z.Off(0, j-1).Vector().Tpsv(uplo, trans, NonUnit, n, bp, 1); err != nil {
					panic(err)
				}
			}

		} else if itype == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**T*y
			if upper {
				trans = Trans
			} else {
				trans = NoTrans
			}

			for j = 1; j <= m; j++ {
				if err = z.Off(0, j-1).Vector().Tpmv(uplo, trans, NonUnit, n, bp, 1); err != nil {
					panic(err)
				}
			}
		}
	}

	return
}
