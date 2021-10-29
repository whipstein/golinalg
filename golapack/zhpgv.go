package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhpgv computes all the eigenvalues and, optionally, the eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be Hermitian, stored in packed format,
// and B is also positive definite.
func Zhpgv(itype int, jobz byte, uplo mat.MatUplo, n int, ap, bp *mat.CVector, w *mat.Vector, z *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (info int, err error) {
	var upper, wantz bool
	var trans mat.MatTrans
	var j, neig int

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper

	if itype < 1 || itype > 3 {
		err = fmt.Errorf("itype < 1 || itype > 3: itype=%v", itype)
	} else if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(upper || uplo == Lower) {
		err = fmt.Errorf("!(upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zhpgv", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	if info, err = Zpptrf(uplo, n, bp); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	if err = Zhpgst(itype, uplo, n, ap, bp); err != nil {
		panic(err)
	}
	if info, err = Zhpev(jobz, uplo, n, ap, w, z, work, rwork); err != nil {
		panic(err)
	}

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		neig = n
		if info > 0 {
			neig = info - 1
		}
		if itype == 1 || itype == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y
			if upper {
				trans = NoTrans
			} else {
				trans = ConjTrans
			}

			for j = 1; j <= neig; j++ {
				if err = goblas.Ztpsv(uplo, trans, NonUnit, n, bp, z.CVector(0, j-1, 1)); err != nil {
					panic(err)
				}
			}

		} else if itype == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = ConjTrans
			} else {
				trans = NoTrans
			}

			for j = 1; j <= neig; j++ {
				if err = goblas.Ztpmv(uplo, trans, NonUnit, n, bp, z.CVector(0, j-1, 1)); err != nil {
					panic(err)
				}
			}
		}
	}

	return
}
