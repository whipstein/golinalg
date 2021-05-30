package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zppt03 computes the residual for a Hermitian packed matrix times its
// inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Zppt03(uplo byte, n *int, a, ainv *mat.CVector, work *mat.CMatrix, ldwork *int, rwork *mat.Vector, rcond, resid *float64) {
	var cone, czero complex128
	var ainvnm, anorm, eps, one, zero float64
	var i, j, jj int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlanhp('1', uplo, n, a, rwork)
	ainvnm = golapack.Zlanhp('1', uplo, n, ainv, rwork)
	if anorm <= zero || ainvnm <= zero {
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
			goblas.Zcopy(&j, ainv.Off(jj-1), func() *int { y := 1; return &y }(), work.CVector(0, j+1-1), func() *int { y := 1; return &y }())
			for i = 1; i <= j-1; i++ {
				work.Set(j-1, i+1-1, ainv.GetConj(jj+i-1-1))
			}
			jj = jj + j
		}
		jj = (((*n)-1)*(*n))/2 + 1
		for i = 1; i <= (*n)-1; i++ {
			work.Set((*n)-1, i+1-1, ainv.GetConj(jj+i-1-1))
		}

		//        Multiply by A
		for j = 1; j <= (*n)-1; j++ {
			goblas.Zhpmv(Upper, n, toPtrc128(-cone), a, work.CVector(0, j+1-1), func() *int { y := 1; return &y }(), &czero, work.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
		goblas.Zhpmv(Upper, n, toPtrc128(-cone), a, ainv.Off(jj-1), func() *int { y := 1; return &y }(), &czero, work.CVector(0, (*n)-1), func() *int { y := 1; return &y }())

		//     UPLO = 'L':
		//     Copy the trailing N-1 x N-1 submatrix of AINV to WORK(1:N,1:N-1)
		//     and multiply by A, moving each column to the right.
	} else {
		//        Copy AINV
		for i = 1; i <= (*n)-1; i++ {
			work.Set(0, i-1, ainv.GetConj(i+1-1))
		}
		jj = (*n) + 1
		for j = 2; j <= (*n); j++ {
			goblas.Zcopy(toPtr((*n)-j+1), ainv.Off(jj-1), func() *int { y := 1; return &y }(), work.CVector(j-1, j-1-1), func() *int { y := 1; return &y }())
			for i = 1; i <= (*n)-j; i++ {
				work.Set(j-1, j+i-1-1, ainv.GetConj(jj+i-1))
			}
			jj = jj + (*n) - j + 1
		}

		//        Multiply by A
		for j = (*n); j >= 2; j-- {
			goblas.Zhpmv(Lower, n, toPtrc128(-cone), a, work.CVector(0, j-1-1), func() *int { y := 1; return &y }(), &czero, work.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
		goblas.Zhpmv(Lower, n, toPtrc128(-cone), a, ainv.Off(0), func() *int { y := 1; return &y }(), &czero, work.CVector(0, 0), func() *int { y := 1; return &y }())

	}

	//     Add the identity matrix to WORK .
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+cone)
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Zlange('1', n, n, work, ldwork, rwork)

	(*resid) = (((*resid) * (*rcond)) / eps) / float64(*n)
}