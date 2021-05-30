package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgeqls Solve the least squares problem
//     min || A*X - B ||
// using the QL factorization
//     A = Q*L
// computed by DGEQLF.
func Dgeqls(m, n, nrhs *int, a *mat.Matrix, lda *int, tau *mat.Vector, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var one float64

	one = 1.0

	//     Test the input arguments.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) > (*m) {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -8
	} else if (*lwork) < 1 || (*lwork) < (*nrhs) && (*m) > 0 && (*n) > 0 {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQLS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     B := Q' * B
	golapack.Dormql('L', 'T', m, nrhs, n, a, lda, tau, b, ldb, work, lwork, info)

	//     Solve L*X = B(m-n+1:m,:)
	goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.NonUnit, n, nrhs, &one, a.Off((*m)-(*n)+1-1, 0), lda, b.Off((*m)-(*n)+1-1, 0), ldb)
}
