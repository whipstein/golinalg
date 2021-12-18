package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dspgvd computes all the eigenvalues, and optionally, the eigenvectors
// of a real generalized symmetric-definite eigenproblem, of the form
// A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
// B are assumed to be symmetric, stored in packed format, and B is also
// positive definite.
// If eigenvectors are desired, it uses a divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dspgvd(itype int, jobz byte, uplo mat.MatUplo, n int, ap, bp, w *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery, upper, wantz bool
	var trans mat.MatTrans
	var j, liwmin, lwmin, neig int

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
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err == nil {
		if n <= 1 {
			liwmin = 1
			lwmin = 1
		} else {
			if wantz {
				liwmin = 3 + 5*n
				lwmin = 1 + 6*n + 2*pow(n, 2)
			} else {
				liwmin = 1
				lwmin = 2 * n
			}
		}
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin
		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dspgvd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form a Cholesky factorization of BP.
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
	if info, err = Dspevd(jobz, uplo, n, ap, w, z, work, lwork, iwork, liwork); err != nil {
		panic(err)
	}
	lwmin = max(lwmin, int(work.Get(0)))
	liwmin = max(liwmin, (*iwork)[0])

	if wantz {
		//        Backtransform eigenvectors to the original problem.
		neig = n
		if info > 0 {
			neig = info - 1
		}
		if itype == 1 || itype == 2 {
			//           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
			//           backtransform eigenvectors: x = inv(L)**T *y or inv(U)*y
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
			//           backtransform eigenvectors: x = L*y or U**T *y
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

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
