package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspgv computes all the eigenvalues and, optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be symmetric, stored in packed format,
// and B is also positive definite.
func Dspgv(itype int, jobz byte, uplo mat.MatUplo, n int, ap, bp, w *mat.Vector, z *mat.Matrix, work *mat.Vector) (info int, err error) {
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
		gltest.Xerbla2("Dspgv", err)
		return
	}

	//     Quick return if possible
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
	if info, err = Dspev(jobz, uplo, n, ap, w, z, work); err != nil {
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
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = NoTrans
			} else {
				trans = Trans
			}

			for j = 1; j <= neig; j++ {
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

			for j = 1; j <= neig; j++ {
				if err = z.Off(0, j-1).Vector().Tpmv(uplo, trans, NonUnit, n, bp, 1); err != nil {
					panic(err)
				}
			}
		}
	}

	return
}
