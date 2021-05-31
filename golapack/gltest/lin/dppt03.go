package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dppt03 computes the residual for a symmetric packed matrix times its
// inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Dppt03(uplo byte, n *int, a, ainv *mat.Vector, work *mat.Matrix, ldwork *int, rwork *mat.Vector, rcond, resid *float64) {
	var ainvnm, anorm, eps, one, zero float64
	var i, j, jj int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Dlansp('1', uplo, n, a, rwork)
	ainvnm = golapack.Dlansp('1', uplo, n, ainv, rwork)
	if anorm <= zero || ainvnm == zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     UPLO = 'U':
	//     Copy the leading N-1 x N-1 submatrix of AINV to WORK(1:N,2:N) and
	//     expand it to a full matrix, then multiply by A one column at a
	//     time, moving the result one column to the left.
	if uplo == 'U' {
		//        Copy AINV
		jj = 1
		for j = 1; j <= (*n)-1; j++ {
			goblas.Dcopy(&j, ainv.Off(jj-1), toPtr(1), work.Vector(0, j+1-1), toPtr(1))
			goblas.Dcopy(toPtr(j-1), ainv.Off(jj-1), toPtr(1), work.Vector(j-1, 1), ldwork)
			jj = jj + j
		}
		jj = (((*n)-1)*(*n))/2 + 1
		goblas.Dcopy(toPtr((*n)-1), ainv.Off(jj-1), toPtr(1), work.Vector((*n)-1, 1), ldwork)

		//        Multiply by A
		for j = 1; j <= (*n)-1; j++ {
			goblas.Dspmv(mat.Upper, n, toPtrf64(-one), a, work.Vector(0, j+1-1), toPtr(1), &zero, work.Vector(0, j-1), toPtr(1))
		}
		goblas.Dspmv(mat.Upper, n, toPtrf64(-one), a, ainv.Off(jj-1), toPtr(1), &zero, work.Vector(0, (*n)-1), toPtr(1))

		//     UPLO = 'L':
		//     Copy the trailing N-1 x N-1 submatrix of AINV to WORK(1:N,1:N-1)
		//     and multiply by A, moving each column to the right.
	} else {
		//        Copy AINV
		goblas.Dcopy(toPtr((*n)-1), ainv.Off(1), toPtr(1), work.Vector(0, 0), ldwork)
		jj = (*n) + 1
		for j = 2; j <= (*n); j++ {
			goblas.Dcopy(toPtr((*n)-j+1), ainv.Off(jj-1), toPtr(1), work.Vector(j-1, j-1-1), toPtr(1))
			goblas.Dcopy(toPtr((*n)-j), ainv.Off(jj+1-1), toPtr(1), work.Vector(j-1, j-1), ldwork)
			jj = jj + (*n) - j + 1
		}

		//        Multiply by A
		for j = (*n); j >= 2; j-- {
			goblas.Dspmv(mat.Lower, n, toPtrf64(-one), a, work.Vector(0, j-1-1), toPtr(1), &zero, work.Vector(0, j-1), toPtr(1))
		}
		goblas.Dspmv(mat.Lower, n, toPtrf64(-one), a, ainv.Off(0), toPtr(1), &zero, work.Vector(0, 0), toPtr(1))
	}

	//     Add the identity matrix to WORK .
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+one)
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Dlange('1', n, n, work, ldwork, rwork)

	(*resid) = (((*resid) * (*rcond)) / eps) / float64(*n)
}
