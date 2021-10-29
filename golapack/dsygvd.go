package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsygvd computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be symmetric and B is also positive definite.
// If eigenvectors are desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dsygvd(itype int, jobz byte, uplo mat.MatUplo, n int, a, b *mat.Matrix, w, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery, upper, wantz bool
	var trans mat.MatTrans
	var one float64
	var liopt, liwmin, lopt, lwmin int

	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1 || liwork == -1)

	if itype < 1 || itype > 3 {
		err = fmt.Errorf("itype < 1 || itype > 3: itype=%v", itype)
	} else if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
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
		work.Set(0, float64(lopt))
		(*iwork)[0] = liopt

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dsygvd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
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
	if info, err = Dsyevd(jobz, uplo, n, a, w, work, lwork, iwork, liwork); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.Get(0)))
	liopt = max(liopt, (*iwork)[0])

	if wantz && info == 0 {
		//        Backtransform eigenvectors to the original problem.
		if itype == 1 || itype == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
			if upper {
				trans = NoTrans
			} else {
				trans = Trans
			}

			if err = goblas.Dtrsm(Left, uplo, trans, NonUnit, n, n, one, b, a); err != nil {
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

			if err = goblas.Dtrmm(Left, uplo, trans, NonUnit, n, n, one, b, a); err != nil {
				panic(err)
			}
		}
	}

	work.Set(0, float64(lopt))
	(*iwork)[0] = liopt

	return
}
