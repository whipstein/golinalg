package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhegv2stage computes all the eigenvalues, and optionally, the eigenvectors
// of a complex generalized Hermitian-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
// Here A and B are assumed to be Hermitian and B is also
// positive definite.
// This routine use the 2stage technique for the reduction to tridiagonal
// which showed higher performance on recent architecture and for large
// sizes N>2000.
func Zhegv2stage(itype int, jobz byte, uplo mat.MatUplo, n int, a, b *mat.CMatrix, w *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector) (info int, err error) {
	var lquery, upper, wantz bool
	var trans mat.MatTrans
	var one complex128
	var ib, kd, lhtrd, lwmin, lwtrd, neig int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1)

	if itype < 1 || itype > 3 {
		err = fmt.Errorf("itype < 1 || itype > 3: itype=%v", itype)
	} else if jobz != 'N' {
		err = fmt.Errorf("jobz != 'N': jobz='%c'", jobz)
	} else if !(upper || uplo == Lower) {
		err = fmt.Errorf("!(upper || uplo == Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}

	if err == nil {
		kd = Ilaenv2stage(1, "Zhetrd2stage", []byte{jobz}, n, -1, -1, -1)
		ib = Ilaenv2stage(2, "Zhetrd2stage", []byte{jobz}, n, kd, -1, -1)
		lhtrd = Ilaenv2stage(3, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
		lwtrd = Ilaenv2stage(4, "Zhetrd2stage", []byte{jobz}, n, kd, ib, -1)
		lwmin = n + lhtrd + lwtrd
		work.SetRe(0, float64(lwmin))

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zhegv2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form a Cholesky factorization of B.
	if info, err = Zpotrf(uplo, n, b); err != nil {
		panic(err)
	}
	if info != 0 {
		info = n + info
		return
	}

	//     Transform problem to standard eigenvalue problem and solve.
	if err = Zhegst(itype, uplo, n, a, b); err != nil {
		panic(err)
	}
	if info, err = Zheev2stage(jobz, uplo, n, a, w, work, lwork, rwork); err != nil {
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

			if err = goblas.Ztrsm(Left, uplo, trans, NonUnit, n, neig, one, b, a); err != nil {
				panic(err)
			}

		} else if itype == 3 {
			//           For B*A*x=(lambda)*x;
			//           backtransform eigenvectors: x = L*y or U**H *y
			if upper {
				trans = ConjTrans
			} else {
				trans = NoTrans
			}

			if err = goblas.Ztrmm(Left, uplo, trans, NonUnit, n, neig, one, b, a); err != nil {
				panic(err)
			}
		}
	}

	work.SetRe(0, float64(lwmin))

	return
}
