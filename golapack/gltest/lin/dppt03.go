package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dppt03 computes the residual for a symmetric packed matrix times its
// inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func dppt03(uplo mat.MatUplo, n int, a, ainv *mat.Vector, work *mat.Matrix, rwork *mat.Vector) (rcond, resid float64) {
	var ainvnm, anorm, eps, one, zero float64
	var i, j, jj int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if n <= 0 {
		rcond = one
		resid = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansp('1', uplo, n, a, rwork)
	ainvnm = golapack.Dlansp('1', uplo, n, ainv, rwork)
	if anorm <= zero || ainvnm == zero {
		rcond = zero
		resid = one / eps
		return
	}
	rcond = (one / anorm) / ainvnm

	//     UPLO = 'U':
	//     Copy the leading N-1 x N-1 submatrix of AINV to WORK(1:N,2:N) and
	//     expand it to a full matrix, then multiply by A one column at a
	//     time, moving the result one column to the left.
	if uplo == Upper {
		//        Copy AINV
		jj = 1
		for j = 1; j <= n-1; j++ {
			work.Off(0, j).Vector().Copy(j, ainv.Off(jj-1), 1, 1)
			work.Off(j-1, 1).Vector().Copy(j-1, ainv.Off(jj-1), 1, work.Rows)
			jj = jj + j
		}
		jj = ((n-1)*n)/2 + 1
		work.Off(n-1, 1).Vector().Copy(n-1, ainv.Off(jj-1), 1, work.Rows)

		//        Multiply by A
		for j = 1; j <= n-1; j++ {
			if err = work.Off(0, j-1).Vector().Spmv(Upper, n, -one, a, work.Off(0, j).Vector(), 1, zero, 1); err != nil {
				panic(err)
			}
		}
		if err = work.Off(0, n-1).Vector().Spmv(Upper, n, -one, a, ainv.Off(jj-1), 1, zero, 1); err != nil {
			panic(err)
		}

		//     UPLO = 'L':
		//     Copy the trailing N-1 x N-1 submatrix of AINV to WORK(1:N,1:N-1)
		//     and multiply by A, moving each column to the right.
	} else {
		//        Copy AINV
		work.Off(0, 0).Vector().Copy(n-1, ainv.Off(1), 1, work.Rows)
		jj = n + 1
		for j = 2; j <= n; j++ {
			work.Off(j-1, j-1-1).Vector().Copy(n-j+1, ainv.Off(jj-1), 1, 1)
			work.Off(j-1, j-1).Vector().Copy(n-j, ainv.Off(jj), 1, work.Rows)
			jj = jj + n - j + 1
		}

		//        Multiply by A
		for j = n; j >= 2; j-- {
			if err = work.Off(0, j-1).Vector().Spmv(Lower, n, -one, a, work.Off(0, j-1-1).Vector(), 1, zero, 1); err != nil {
				panic(err)
			}
		}
		if err = work.Off(0, 0).Vector().Spmv(Lower, n, -one, a, ainv, 1, zero, 1); err != nil {
			panic(err)
		}
	}

	//     Add the identity matrix to WORK .
	for i = 1; i <= n; i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+one)
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	resid = golapack.Dlange('1', n, n, work, rwork)

	resid = ((resid * rcond) / eps) / float64(n)

	return
}
